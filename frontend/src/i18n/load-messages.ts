import type { AbstractIntlMessages } from "next-intl";

import { NAMESPACES, defaultLocale, type Locale, type Namespace } from "./config";

/** A namespace that is still being translated falls back to English. */
async function loadNamespace(
  locale: Locale,
  namespace: Namespace,
): Promise<AbstractIntlMessages> {
  try {
    return (await import(`../../messages/${locale}/${namespace}.json`))
      .default as AbstractIntlMessages;
  } catch (error) {
    const code = (error as { code?: string })?.code;
    const message = (error as { message?: string })?.message ?? "";
    if (code !== "MODULE_NOT_FOUND" && !message.includes("Cannot find module"))
      throw error;
    return (await import(`../../messages/${defaultLocale}/${namespace}.json`))
      .default as AbstractIntlMessages;
  }
}

export async function loadMessages(locale: Locale): Promise<AbstractIntlMessages> {
  const entries = await Promise.all(
    NAMESPACES.map(async (namespace) => {
      const messages = await loadNamespace(locale, namespace);
      return [namespace, messages] as const;
    }),
  );

  return Object.fromEntries(entries);
}
