import { z } from "zod";
import { getOptionalEnvValue } from "@/server/lib/runtime-env";
import { AI_ENGINE_LABELS, type AiEngine } from "@/shared/ai-visibility";
import {
  citationsFrom,
  markdownToText,
  type ParsedAiAnswer,
} from "./dataforseoEvidence";

/**
 * Collecting AI visibility without DataForSEO: engines are answered by any
 * OpenAI-compatible chat-completions endpoint whose models search the web and
 * return URL citations — OpenAI's search models (default), OpenRouter `:online`
 * models, or Perplexity's API. Enable with AI_VISIBILITY_PROVIDER=
 * openai_compatible plus AI_VISIBILITY_API_KEY; AI_VISIBILITY_API_BASE and
 * AI_VISIBILITY_MODEL_<ENGINE> reroute individual pieces.
 */

const OPENAI_COMPATIBLE = "openai_compatible";
const DEFAULT_API_BASE = "https://api.openai.com/v1";
// A live web-search answer can take over a minute; matches the per-answer cap
// the DataForSEO live path documents.
const REQUEST_TIMEOUT_MS = 120_000;
const MAX_ERROR_PAYLOAD_LENGTH = 400;

/**
 * Engines this provider can answer. Google AI Overviews is a Google SERP
 * scrape, not a chat model, so it has no equivalent here and fails its rows.
 */
const MODEL_ENV: Record<Exclude<AiEngine, "google_ai_overview">, string> = {
  chatgpt: "AI_VISIBILITY_MODEL_CHATGPT",
  gemini: "AI_VISIBILITY_MODEL_GEMINI",
  claude: "AI_VISIBILITY_MODEL_CLAUDE",
  perplexity: "AI_VISIBILITY_MODEL_PERPLEXITY",
};
const DEFAULT_MODELS: Partial<
  Record<Exclude<AiEngine, "google_ai_overview">, string>
> = {
  chatgpt: "gpt-5-search-api",
};

export async function aiVisibilityProviderMode(): Promise<
  "dataforseo" | "openai_compatible"
> {
  return (await getOptionalEnvValue("AI_VISIBILITY_PROVIDER")) ===
    OPENAI_COMPATIBLE
    ? OPENAI_COMPATIBLE
    : "dataforseo";
}

/** True when collection can run at all: DataForSEO key or this provider. */
export async function isAiCollectionConfigured(): Promise<boolean> {
  if (await getOptionalEnvValue("DATAFORSEO_API_KEY")) return true;
  return (
    (await aiVisibilityProviderMode()) === OPENAI_COMPATIBLE &&
    !!(await getOptionalEnvValue("AI_VISIBILITY_API_KEY"))
  );
}

async function resolveEngineModel(engine: AiEngine): Promise<string> {
  if (engine === "google_ai_overview")
    throw new Error(
      "Google AI Overviews has no OpenAI-compatible model; it needs DataForSEO.",
    );
  const configured = await getOptionalEnvValue(MODEL_ENV[engine]);
  const model = configured ?? DEFAULT_MODELS[engine];
  if (!model)
    throw new Error(
      `No model configured for the ${AI_ENGINE_LABELS[engine]} engine. Set ${MODEL_ENV[engine]} to a web-search model on your AI_VISIBILITY_API_BASE endpoint (e.g. an OpenRouter ":online" model).`,
    );
  return model;
}

// OpenAI and OpenRouter nest the citation under `url_citation`; some
// OpenAI-compatible gateways mirror the fields at the top level.
const annotationSchema = z.object({
  url_citation: z
    .object({ url: z.string(), title: z.string().nullish() })
    .nullish(),
  url: z.string().nullish(),
  title: z.string().nullish(),
});
const responseSchema = z.object({
  choices: z
    .array(
      z.object({
        message: z.object({
          content: z.string().nullish(),
          annotations: z.array(z.unknown()).nullish(),
        }),
      }),
    )
    .min(1),
  // Perplexity chat-completions fields alongside choices.
  citations: z.array(z.string()).nullish(),
  search_results: z
    .array(z.object({ url: z.string(), title: z.string().nullish() }))
    .nullish(),
});

/** Citation items in the shape `citationsFrom` normalizes. */
function citedSources(
  result: z.infer<typeof responseSchema>,
): { url: string; title?: string | null }[] {
  const annotations = (result.choices[0]?.message.annotations ?? [])
    .map((annotation) => annotationSchema.safeParse(annotation))
    .filter((parsed) => parsed.success)
    .map((parsed) => parsed.data);
  return [
    ...annotations.flatMap((annotation) =>
      annotation.url_citation
        ? [
            {
              url: annotation.url_citation.url,
              title: annotation.url_citation.title,
            },
          ]
        : [],
    ),
    ...annotations.flatMap((annotation) =>
      annotation.url ? [{ url: annotation.url, title: annotation.title }] : [],
    ),
    ...(result.search_results ?? []),
    ...(result.citations ?? []).map((url) => ({ url })),
  ];
}

/**
 * One answer through the configured endpoint. Billed by the endpoint's owner
 * (per-token, per-query, or self-hosted), so like the DataForSEO live path it
 * never retries: the caller fails only this answer and keeps the rest.
 */
export async function collectProviderAnswer(input: {
  engine: AiEngine;
  prompt: string;
  /** ISO country code; nudges the web search toward the tracker's market. */
  countryCode: string;
}): Promise<ParsedAiAnswer> {
  const model = await resolveEngineModel(input.engine);
  const apiKey = await getOptionalEnvValue("AI_VISIBILITY_API_KEY");
  if (!apiKey)
    throw new Error(
      "AI_VISIBILITY_API_KEY is required when AI_VISIBILITY_PROVIDER=openai_compatible.",
    );
  const base =
    (await getOptionalEnvValue("AI_VISIBILITY_API_BASE")) ?? DEFAULT_API_BASE;
  const label = AI_ENGINE_LABELS[input.engine];
  let response: Response;
  try {
    response = await fetch(`${base.replace(/\/+$/, "")}/chat/completions`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "system",
            content: `You are ${label}, a consumer AI assistant, answering for an AI-visibility study. Search the web, answer in concise markdown with inline citations, and prefer results relevant to ${input.countryCode}.`,
          },
          { role: "user", content: input.prompt },
        ],
      }),
    });
  } catch {
    throw new Error(
      `The ${label} answer request timed out or could not reach ${base}.`,
    );
  }
  if (!response.ok) {
    const detail = (await response.text()).slice(0, MAX_ERROR_PAYLOAD_LENGTH);
    throw new Error(
      `The ${label} answer request failed with HTTP ${response.status}: ${detail}`,
    );
  }
  const body: unknown = await response.json();
  const parsed = responseSchema.safeParse(body);
  if (!parsed.success) throw new Error("Unreadable answer from the provider.");
  const markdown = parsed.data.choices[0]?.message.content ?? null;
  const answerText = markdown ? markdownToText(markdown) : "";
  return {
    answerMarkdown: answerText.trim() ? markdown : null,
    answerText: answerText.trim() ? answerText : "",
    citations: citationsFrom(citedSources(parsed.data)),
    collectedAt: null,
  };
}
