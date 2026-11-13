"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useLocale, useTranslations } from "next-intl";

import { Button } from "@/components/ui/button";
import { intlLocale } from "@/i18n/config";
import { cn } from "@/lib/utils";

const POLL_INTERVAL_MS = 30_000;

export interface AdminLiveStatsData {
  users: number;
  admins: number;
  tasks: number;
  completed: number;
  processing: number;
  queued: number;
}

interface StatsPayload {
  stats?: AdminLiveStatsData;
  updatedAt?: string;
}

export function AdminLiveStats({
  initialStats,
  initialUpdatedAt,
}: {
  initialStats: AdminLiveStatsData;
  initialUpdatedAt: string;
}) {
  const t = useTranslations("admin");
  const tLibrary = useTranslations("library");
  const tStatus = useTranslations("status");
  const tSocial = useTranslations("social");
  const dateLocale = intlLocale(useLocale());

  const [stats, setStats] = useState<AdminLiveStatsData>(initialStats);
  const [updatedAt, setUpdatedAt] = useState<string>(initialUpdatedAt);
  const [loading, setLoading] = useState(false);
  const [failed, setFailed] = useState(false);
  // Server and browser time zones differ: the clock text is client-only.
  const [isHydrated, setIsHydrated] = useState(false);
  useEffect(() => setIsHydrated(true), []);

  const mountedRef = useRef(true);
  const inFlightRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    const controller = new AbortController();
    abortRef.current = controller;
    setLoading(true);
    setFailed(false);
    try {
      const response = await fetch("/api/admin/stats", {
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`stats request failed: ${response.status}`);
      const payload = (await response.json()) as StatsPayload;
      if (!payload.stats) throw new Error("stats payload missing");
      if (!mountedRef.current) return;
      setStats(payload.stats);
      setUpdatedAt(payload.updatedAt ?? new Date().toISOString());
    } catch {
      if (mountedRef.current && !controller.signal.aborted) setFailed(true);
    } finally {
      if (abortRef.current === controller) abortRef.current = null;
      inFlightRef.current = false;
      if (mountedRef.current) setLoading(false);
    }
  }, []);

  // The polling interval is created once, so it must reach the latest
  // refresh function through a ref instead of a re-created closure.
  const refreshRef = useRef(refresh);
  useEffect(() => {
    refreshRef.current = refresh;
  });

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      abortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;

    const stop = () => {
      if (interval === null) return;
      clearInterval(interval);
      interval = null;
    };

    const start = () => {
      if (interval !== null) return;
      interval = setInterval(() => {
        void refreshRef.current();
      }, POLL_INTERVAL_MS);
    };

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        start();
        void refreshRef.current();
      } else {
        stop();
      }
    };

    // The server already rendered initialStats, so the first poll is 30s later.
    if (document.visibilityState === "visible") start();
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, []);

  const updatedLabel = tStatus("lastUpdated");
  const updatedTime = new Date(updatedAt).toLocaleTimeString(dateLocale, {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const cards = [
    { key: "users", label: t("stats.users"), value: stats.users, live: false },
    { key: "admins", label: t("stats.admins"), value: stats.admins, live: false },
    { key: "tasks", label: t("stats.tasks"), value: stats.tasks, live: false },
    { key: "completed", label: t("stats.completed"), value: stats.completed, live: false },
    { key: "processing", label: t("stats.processing"), value: stats.processing, live: stats.processing > 0 },
    { key: "queued", label: tLibrary("status.queued"), value: stats.queued, live: false },
  ];

  return (
    <section className="space-y-3">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        {cards.map((card) => (
          <div
            key={card.key}
            className={cn("rounded-2xl border bg-background p-4", card.live && "border-brand/40 bg-brand-soft")}
          >
            <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
              {card.live && (
                <span className="size-1.5 rounded-full bg-brand motion-safe:animate-pulse" />
              )}
              {card.label}
            </p>
            <p className="mt-1 font-display text-3xl font-bold tabular-nums">{card.value}</p>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="space-y-1">
          {isHydrated && (
            <p className="text-xs text-muted-foreground">
              <span>{updatedLabel}</span> <span className="tabular-nums">{updatedTime}</span>
            </p>
          )}
          {failed && (
            <p role="alert" className="text-xs text-red-600">
              {tLibrary("somethingWrong")}
            </p>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          disabled={loading}
          aria-label={tSocial("refreshMetrics")}
          onClick={() => {
            void refreshRef.current();
          }}
        >
          {tSocial("refreshMetrics")}
        </Button>
      </div>
    </section>
  );
}
