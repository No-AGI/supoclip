import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { NextIntlClientProvider } from "next-intl";
import { beforeEach, expect, it, vi } from "vitest";

import { LlmSettingsPanel } from "./llm-settings-panel";
import { LlmConnections } from "./llm-connections";

const settings = {
  primary: "google-gla:gemini-3-flash-preview",
  fallbacks: ["google-gla:gemini-2.5-flash"],
  chain: [
    "google-gla:gemini-3-flash-preview",
    "google-gla:gemini-2.5-flash",
  ],
  providers: ["google", "google-gla"],
  keys: {
    google: [{ fingerprint: "…4321·abcdef", source: "admin" }],
  },
};

beforeEach(() => {
  vi.resetAllMocks();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      if (url === "/api/admin/llm") return Response.json(settings);
      if (url.endsWith("/chain") && init?.method === "PATCH") {
        return Response.json({ chain: settings.chain });
      }
      if (url.endsWith("/test/chain")) {
        return Response.json({
          ok: true,
          seconds: 3.2,
          answered_by: "google-gla:gemini-2.5-flash",
          attempts: [
            {
              model: "google-gla:gemini-3-flash-preview",
              provider: "google-gla",
              key_fingerprint: null,
              seconds: 1.1,
              ok: false,
              kind: "quota",
              error: "You exceeded your current quota",
            },
            {
              model: "google-gla:gemini-2.5-flash",
              provider: "google-gla",
              key_fingerprint: "…4321·abcdef",
              seconds: 2.1,
              ok: true,
              kind: null,
              error: "",
            },
          ],
        });
      }
      return Response.json({});
    }),
  );
});

it("shows the chain of models and the marks of the keys", async () => {
  render(<LlmSettingsPanel />);

  await waitFor(() =>
    expect(
      screen.getByText("google-gla:gemini-3-flash-preview"),
    ).toBeInTheDocument(),
  );
  expect(
    screen.getByText("google-gla:gemini-2.5-flash"),
  ).toBeInTheDocument();
  // The mark of a key is shown, the key itself never is.
  expect(screen.getByText("…4321·abcdef")).toBeInTheDocument();
  expect(screen.queryByText(/4321·abcdef/)).not.toBeNull();
});

it("saves the chain after a model was added", async () => {
  render(<LlmSettingsPanel />);
  await waitFor(() => expect(fetch).toHaveBeenCalled());

  fireEvent.change(
    screen.getByPlaceholderText(/provider:model/i),
    { target: { value: "google-gla:gemini-2.5-flash-lite" } },
  );
  fireEvent.click(screen.getByRole("button", { name: /add a model/i }));
  fireEvent.click(screen.getByRole("button", { name: /^save$/i }));

  await waitFor(() => {
    const call = (fetch as unknown as { mock: { calls: unknown[][] } }).mock.calls
      .find((entry) => String(entry[0]).endsWith("/chain"));
    expect(call).toBeTruthy();
    expect(String((call?.[1] as { body: string }).body)).toContain(
      "gemini-2.5-flash-lite",
    );
  });
});

it("shows every attempt of the chain", async () => {
  render(<LlmSettingsPanel />);
  await waitFor(() => expect(fetch).toHaveBeenCalled());

  fireEvent.click(screen.getByRole("button", { name: /run the chain/i }));

  await waitFor(() =>
    expect(
      screen.getByText("You exceeded your current quota"),
    ).toBeInTheDocument(),
  );
  expect(screen.getByText("Answered")).toBeInTheDocument();
  // The model of the chain and the model of the attempt are both listed.
  expect(
    screen.getAllByText("google-gla:gemini-2.5-flash").length,
  ).toBeGreaterThan(0);
});

it("tests a custom connection explicitly, then adds it to the unsaved fallback chain", async () => {
  const calls: Array<{ url: string; init?: RequestInit }> = [];
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    if (url === "/api/admin/llm") return Response.json(settings);
    if (url === "/api/admin/llm/connections") {
      return Response.json({ connections: [{ id: "conn-1", name: "Groq", base_url: "https://api.groq.com/openai/v1", model: "llama-test", key_fingerprint: "…abcd" }] });
    }
    if (url.endsWith("/conn-1/test")) return Response.json({ id: "conn-1", ok: true, seconds: 0.4, model: "conn-1:llama-test" });
    if (url.endsWith("/chain") && init?.method === "PATCH") return Response.json({});
    return Response.json({});
  }));
  render(<LlmSettingsPanel />);
  const test = await screen.findByRole("button", { name: /test connection/i });
  expect(screen.getByText(/…abcd/)).toBeInTheDocument();
  fireEvent.click(test);
  fireEvent.click(await screen.findByRole("button", { name: /add to fallback chain/i }));
  expect(screen.getByText("conn-1:llama-test")).toBeInTheDocument();

  // Adding is only an edit to the current chain until Save is clicked.
  expect(calls.some(({ url }) => url.endsWith("/chain"))).toBe(false);
  fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
  await waitFor(() => expect(calls.some(({ url }) => url.endsWith("/chain"))).toBe(true));
  const saved = calls.find(({ url }) => url.endsWith("/chain"));
  expect(saved?.init?.body).toContain("conn-1:llama-test");
});

it("uses inline Russian labels for custom connections when the locale catalog has no new keys", async () => {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    if (url === "/api/admin/llm/connections") return Response.json({ connections: [{ id: "ru-conn", name: "DeepSeek", base_url: "https://api.deepseek.com/v1", model: "deepseek-chat", key_fingerprint: "…abcd" }] });
    if (url.endsWith("/ru-conn/test")) return Response.json({ id: "ru-conn", ok: true, seconds: 0.7, model: "ru-conn:deepseek-chat" });
    return Response.json({});
  }));
  render(
    <NextIntlClientProvider locale="ru" messages={{ admin: { llm: { remove: "Удалить" } } }}>
      <LlmConnections onAddFallback={() => undefined} />
    </NextIntlClientProvider>,
  );
  expect(await screen.findByText("Пользовательские подключения LLM")).toBeInTheDocument();
  expect(screen.getByLabelText("API-ключ")).toHaveAttribute("type", "password");
  fireEvent.click(screen.getByRole("button", { name: "Проверить подключение" }));
  expect(await screen.findByText("Подключение работает, время: 0.7 с")).toBeInTheDocument();
});
