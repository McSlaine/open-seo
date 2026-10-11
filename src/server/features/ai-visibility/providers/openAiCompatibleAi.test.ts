import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  aiVisibilityProviderMode,
  collectProviderAnswer,
  isAiCollectionConfigured,
} from "./openAiCompatibleAi";

const env: Record<string, string | undefined> = {};
const fetchMock = vi.fn();

vi.mock("@/server/lib/runtime-env", () => ({
  getOptionalEnvValue: async (name: string) => env[name],
}));

beforeEach(() => {
  for (const name of Object.keys(env)) delete env[name];
  vi.stubGlobal("fetch", fetchMock);
  fetchMock.mockResolvedValue(
    new Response(JSON.stringify({ choices: [{ message: { content: "" } }] }), {
      status: 200,
    }),
  );
});

function openAiBody() {
  return {
    choices: [
      {
        message: {
          role: "assistant",
          content:
            "Try **Ahrefs** [1](https://openseo.so/pricing?utm_source=x).",
          annotations: [
            {
              type: "url_citation",
              url_citation: {
                url: "https://openseo.so/pricing?utm_source=x",
                title: "Pricing",
                start_index: 12,
                end_index: 14,
              },
            },
          ],
        },
      },
    ],
  };
}

describe("OpenAI-compatible provider", () => {
  it("sends the engine's model and prompt to the configured endpoint", async () => {
    env.AI_VISIBILITY_API_KEY = "key";
    let sentUrl = "";
    let sentInit: RequestInit | undefined;
    fetchMock.mockImplementationOnce((url: string, init: RequestInit) => {
      sentUrl = url;
      sentInit = init;
      return Promise.resolve(
        new Response(JSON.stringify(openAiBody()), { status: 200 }),
      );
    });

    await collectProviderAnswer({
      engine: "chatgpt",
      prompt: "Which SEO tools?",
      countryCode: "US",
    });

    if (!sentInit) throw new Error("the request was not sent");
    const body = sentInit.body;
    if (typeof body !== "string") throw new Error("body should be JSON text");
    expect(sentUrl).toBe("https://api.openai.com/v1/chat/completions");
    expect(sentInit.headers).toMatchObject({ Authorization: "Bearer key" });
    expect(body).toContain('"model":"gpt-5-search-api"');
    expect(body).toContain('"content":"Which SEO tools?"');
  });

  it("reads nested citation annotations and strips tracking parameters", async () => {
    env.AI_VISIBILITY_API_KEY = "key";
    fetchMock.mockResolvedValueOnce(
      new Response(JSON.stringify(openAiBody()), { status: 200 }),
    );

    const answer = await collectProviderAnswer({
      engine: "chatgpt",
      prompt: "Which SEO tools?",
      countryCode: "US",
    });

    expect(answer.answerMarkdown).toContain("Ahrefs");
    expect(answer.citations).toEqual([
      {
        url: "https://openseo.so/pricing",
        domain: "openseo.so",
        title: "Pricing",
        position: 1,
      },
    ]);
  });

  it("reads Perplexity-style top-level citations, de-duplicated", async () => {
    env.AI_VISIBILITY_API_KEY = "key";
    fetchMock.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          choices: [{ message: { content: "Answer." } }],
          search_results: [{ url: "https://b.com/", title: "B" }],
          citations: ["https://a.com/x", "https://a.com/x"],
        }),
        { status: 200 },
      ),
    );

    const answer = await collectProviderAnswer({
      engine: "chatgpt",
      prompt: "q",
      countryCode: "US",
    });

    expect(answer.citations).toMatchObject([
      { url: "https://b.com/", domain: "b.com", position: 1 },
      { url: "https://a.com/x", domain: "a.com", position: 2 },
    ]);
  });

  it("fails with the env var to set when the engine has no model", async () => {
    env.AI_VISIBILITY_API_KEY = "key";

    await expect(
      collectProviderAnswer({
        engine: "gemini",
        prompt: "q",
        countryCode: "US",
      }),
    ).rejects.toThrow("AI_VISIBILITY_MODEL_GEMINI");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("surfaces provider HTTP errors without retrying", async () => {
    env.AI_VISIBILITY_API_KEY = "key";
    fetchMock.mockResolvedValueOnce(new Response("denied", { status: 401 }));

    await expect(
      collectProviderAnswer({
        engine: "chatgpt",
        prompt: "q",
        countryCode: "US",
      }),
    ).rejects.toThrow("HTTP 401");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("is configured with a DataForSEO key or the provider key", async () => {
    expect(await aiVisibilityProviderMode()).toBe("dataforseo");
    expect(await isAiCollectionConfigured()).toBe(false);

    env.AI_VISIBILITY_PROVIDER = "openai_compatible";
    expect(await aiVisibilityProviderMode()).toBe("openai_compatible");
    expect(await isAiCollectionConfigured()).toBe(false);

    env.AI_VISIBILITY_API_KEY = "key";
    expect(await isAiCollectionConfigured()).toBe(true);

    delete env.AI_VISIBILITY_PROVIDER;
    delete env.AI_VISIBILITY_API_KEY;
    env.DATAFORSEO_API_KEY = "abc";
    expect(await isAiCollectionConfigured()).toBe(true);
  });
});
