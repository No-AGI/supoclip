import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createTranslator } from "next-intl";
import { describe, expect, it, vi } from "vitest";
import { NAMESPACES } from "./config";
import { timeAgo } from "@/lib/generations";

type Messages = { [key: string]: string | Messages };
const read = (locale: string, ns: string): Messages => JSON.parse(readFileSync(join(process.cwd(), "messages", locale, `${ns}.json`), "utf8"));
function flatten(messages: Messages, prefix = ""): [string, string][] {
  return Object.entries(messages).flatMap(([key, value]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return typeof value === "string" ? [[path, value] as [string, string]] : flatten(value, path);
  });
}

describe("Russian interface", () => {
  it("formats every Russian message, preserving rich-text tags and ICU arguments", () => {
    for (const ns of NAMESPACES) {
      const english = new Map(flatten(read("en", ns)));
      const messages = read("ru", ns);
      const t = createTranslator({ locale: "ru", messages: messages as Record<string, string>, onError: error => { throw error; } });
      for (const [key, value] of flatten(messages)) {
        const source = english.get(key)!;
        const tags = (text: string) => [...text.matchAll(/<\/?([A-Za-z][\w]*)\s*\/?\s*>/g)].map(match => match[0]).sort();
        expect(tags(value), `${ns}.${key}`).toEqual(tags(source));
        const values: Record<string, string | number | ((chunks: string) => string)> = {};
        for (const match of value.matchAll(/\{\s*([\w]+)\s*([,}])/g)) values[match[1]] = match[2] === "," && /\{\s*[^,]+,\s*(plural|selectordinal|number|date|time)/.test(value.slice(match.index)) ? 2 : "пример";
        for (const match of value.matchAll(/<([A-Za-z][\w]*)>/g)) values[match[1]] = chunks => chunks;
        // Render rich messages as plain text to exercise the actual ICU compiler.
        expect(() => t.markup(key, values)).not.toThrow();
        if (source.includes(", plural,")) {
          expect(value, `${ns}.${key}`).toContain("few {");
          expect(value, `${ns}.${key}`).toContain("many {");
        }
      }
    }
  });

  it("uses Russian plural forms for counts including teens and fractions", () => {
    const t = createTranslator({ locale: "ru", messages: { duration: { hours: (read("ru", "common").duration as Messages).hours as "{count, plural, other {#}}" } } });
    for (const [count, expected] of [[1, "1 час"], [2, "2 часа"], [5, "5 часов"], [11, "11 часов"], [21, "21 час"], [22, "22 часа"], [25, "25 часов"], [1.5, "1,5 часа"]] as const) {
      expect(t("duration.hours", { count })).toBe(expected);
    }
  });

  it("uses Russian relative dates including the sub-minute label", () => {
    vi.useFakeTimers();
    try {
      vi.setSystemTime(new Date("2026-11-12T12:00:00Z"));
      expect(timeAgo("2026-11-12T11:59:50Z", "ru")).toBe("только что");
      expect(timeAgo("2026-11-12T11:58:00Z", "ru")).toBe("2 минуты назад");
    } finally { vi.useRealTimers(); }
  });
});
