"use client";

import { useEffect, useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

type Connection = { id: string; name: string; base_url: string; model: string; key_fingerprint: string };
type TestResult = { id: string; ok: boolean; seconds: number; model?: string; error?: string };
const PRESETS: Record<string, { url: string; model: string }> = {
  Groq: { url: "https://api.groq.com/openai/v1", model: "llama-3.3-70b-versatile" },
  Together: { url: "https://api.together.xyz/v1", model: "meta-llama/Llama-3.3-70B-Instruct-Turbo" },
  Fireworks: { url: "https://api.fireworks.ai/inference/v1", model: "accounts/fireworks/models/llama-v3p3-70b-instruct" },
  DeepSeek: { url: "https://api.deepseek.com/v1", model: "deepseek-chat" },
};

export function LlmConnections({ onAddFallback }: { onAddFallback: (model: string) => void }) {
  const t = useTranslations("admin.llm");
  const locale = useLocale();
  const ru: Record<string, string> = {
    connectionsTitle: "Пользовательские подключения LLM",
    connectionsDescription: "Добавьте провайдера с поддержкой OpenAI. Ключи хранятся безопасно и больше не отображаются.",
    connectionsLoadFailed: "Не удалось загрузить подключения",
    connectionSaveFailed: "Не удалось сохранить подключение",
    connectionDeleteFailed: "Не удалось удалить подключение",
    connectionTestFailed: "Не удалось проверить подключение",
    connectionPreset: "Провайдер",
    connectionCustom: "Свой URL",
    connectionName: "Название подключения",
    connectionBaseUrl: "Базовый URL",
    connectionModel: "Модель",
    connectionApiKey: "API-ключ",
    connectionSave: "Сохранить подключение",
    connectionTest: "Проверить подключение",
    connectionTestOk: "Подключение работает, время: {seconds} с",
    connectionAddFallback: "Добавить в цепочку резервных моделей",
  };
  const label = (key: keyof typeof ru, values?: { seconds: string }) =>
    locale === "ru" ? ru[key].replace("{seconds}", values?.seconds ?? "") : t(key, values);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [preset, setPreset] = useState("Groq");
  const [name, setName] = useState("Groq");
  const [baseUrl, setBaseUrl] = useState(PRESETS.Groq.url);
  const [model, setModel] = useState(PRESETS.Groq.model);
  const [apiKey, setApiKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [tested, setTested] = useState<Record<string, TestResult>>({});

  async function load() {
    const response = await fetch("/api/admin/llm/connections", { cache: "no-store" });
    if (!response.ok) throw new Error(label("connectionsLoadFailed"));
    const data = await response.json();
    setConnections(data.connections ?? []);
  }
  useEffect(() => { void load().catch((e) => setError(e.message)); }, []);

  function selectPreset(value: string) {
    setPreset(value);
    if (value === "custom") { setName(""); setBaseUrl(""); setModel(""); return; }
    const entry = PRESETS[value];
    setName(value); setBaseUrl(entry.url); setModel(entry.model);
  }

  async function create() {
    setBusy(true); setError("");
    try {
      const response = await fetch("/api/admin/llm/connections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, base_url: baseUrl, model, api_key: apiKey }) });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.error || label("connectionSaveFailed"));
      setApiKey(""); await load();
    } catch (e) { setError(e instanceof Error ? e.message : label("connectionSaveFailed")); }
    finally { setBusy(false); }
  }

  async function remove(id: string) {
    setBusy(true); setError("");
    try {
      const response = await fetch(`/api/admin/llm/connections/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!response.ok) { const data = await response.json().catch(() => ({})); throw new Error(data.detail || label("connectionDeleteFailed")); }
      setTested((current) => { const next = { ...current }; delete next[id]; return next; }); await load();
    } catch (e) { setError(e instanceof Error ? e.message : label("connectionDeleteFailed")); }
    finally { setBusy(false); }
  }

  async function test(id: string) {
    setBusy(true); setError("");
    try {
      const response = await fetch(`/api/admin/llm/connections/${encodeURIComponent(id)}/test`, { method: "POST" });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.error || label("connectionTestFailed"));
      setTested((current) => ({ ...current, [id]: data }));
    } catch (e) { setError(e instanceof Error ? e.message : label("connectionTestFailed")); }
    finally { setBusy(false); }
  }

  return <section className="border-t px-5 py-5">
    <h3 className="text-sm font-medium">{label("connectionsTitle")}</h3>
    <p className="mt-1 text-xs text-muted-foreground">{label("connectionsDescription")}</p>
    <div className="mt-3 space-y-2">
      {connections.map((connection) => {
        const result = tested[connection.id];
        return <div key={connection.id} className="rounded-lg border px-3 py-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div><p className="text-sm font-medium">{connection.name}</p><code className="text-xs text-muted-foreground">{connection.model} · {connection.key_fingerprint}</code></div>
            <div className="flex gap-2">
              <Button size="sm" variant="outline" disabled={busy} onClick={() => void test(connection.id)}>{label("connectionTest")}</Button>
              <Button size="sm" variant="ghost" disabled={busy} onClick={() => void remove(connection.id)}>{t("remove")}</Button>
              {result?.ok && <Button size="sm" variant="secondary" onClick={() => onAddFallback(result.model || `${connection.id}:${connection.model}`)}>{label("connectionAddFallback")}</Button>}
            </div>
          </div>
          {result && <p className={`mt-2 text-xs ${result.ok ? "text-emerald-600" : "text-red-600"}`}>{result.ok ? label("connectionTestOk", { seconds: result.seconds.toFixed(1) }) : result.error || label("connectionTestFailed")}</p>}
        </div>;
      })}
    </div>
    <div className="mt-4 grid gap-2 sm:grid-cols-2">
      <label className="text-xs">{label("connectionPreset")}<select className="mt-1 h-9 w-full rounded-md border bg-background px-3 text-sm" value={preset} onChange={(e) => selectPreset(e.target.value)}>{Object.keys(PRESETS).map((item) => <option key={item}>{item}</option>)}<option value="custom">{label("connectionCustom")}</option></select></label>
      <label className="text-xs">{label("connectionName")}<Input className="mt-1 h-9" value={name} onChange={(e) => setName(e.target.value)} /></label>
      <label className="text-xs">{label("connectionBaseUrl")}<Input className="mt-1 h-9" type="url" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} /></label>
      <label className="text-xs">{label("connectionModel")}<Input className="mt-1 h-9" value={model} onChange={(e) => setModel(e.target.value)} /></label>
      <label className="text-xs sm:col-span-2">{label("connectionApiKey")}<Input className="mt-1 h-9" type="password" autoComplete="new-password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} /></label>
    </div>
    <Button className="mt-3" size="sm" disabled={busy || !name.trim() || !baseUrl.trim() || !model.trim() || !apiKey.trim()} onClick={() => void create()}>{label("connectionSave")}</Button>
    {error && <p role="alert" className="mt-2 text-xs text-red-600">{error}</p>}
  </section>;
}
