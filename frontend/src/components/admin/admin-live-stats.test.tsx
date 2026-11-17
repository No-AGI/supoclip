import { act, fireEvent, render, screen } from "@testing-library/react";

import { AdminLiveStats, type AdminLiveStatsData } from "./admin-live-stats";

const INITIAL: AdminLiveStatsData = {
  users: 11,
  admins: 2,
  tasks: 33,
  completed: 22,
  queued: 7,
  processing: 4,
};

const NEXT: AdminLiveStatsData = {
  users: 21,
  admins: 5,
  tasks: 44,
  completed: 55,
  queued: 9,
  processing: 6,
};

const UPDATED_AT = "2026-11-13T00:00:00.000Z";
const originalVisibilityState = document.visibilityState;

function renderStats() {
  return render(
    <AdminLiveStats initialStats={INITIAL} initialUpdatedAt={UPDATED_AT} />,
  );
}

describe("AdminLiveStats", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", vi.fn());
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: originalVisibilityState,
    });
  });

  it("renders the six stats from the server render", () => {
    renderStats();

    expect(screen.getByText("Total users")).toBeInTheDocument();
    expect(screen.getByText("Admins")).toBeInTheDocument();
    expect(screen.getByText("Total tasks")).toBeInTheDocument();
    expect(screen.getByText("Completed tasks")).toBeInTheDocument();
    expect(screen.getByText("Currently processing")).toBeInTheDocument();
    expect(screen.getByText("Queued")).toBeInTheDocument();

    expect(screen.getByText("11")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("33")).toBeInTheDocument();
    expect(screen.getByText("22")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();

    expect(screen.getByText("Last updated:")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Refresh metrics" })).toBeEnabled();
    expect(fetch).not.toHaveBeenCalled();
  });

  it("fetches new numbers when the refresh button is used", async () => {
    let resolveFetch!: (value: Response) => void;
    vi.mocked(fetch).mockImplementation(
      () =>
        new Promise<Response>((resolve) => {
          resolveFetch = resolve;
        }),
    );

    renderStats();
    const button = screen.getByRole("button", { name: "Refresh metrics" });

    await act(async () => {
      fireEvent.click(button);
    });

    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith(
      "/api/admin/stats",
      expect.objectContaining({ cache: "no-store" }),
    );
    expect(button).toBeDisabled();

    await act(async () => {
      resolveFetch(Response.json({ stats: NEXT, updatedAt: UPDATED_AT }));
    });

    expect(button).toBeEnabled();
    expect(screen.getByText("21")).toBeInTheDocument();
    expect(screen.getByText("55")).toBeInTheDocument();
    expect(screen.getByText("9")).toBeInTheDocument();
    expect(screen.queryByText("11")).not.toBeInTheDocument();
    expect(screen.queryByText("22")).not.toBeInTheDocument();
    expect(screen.queryByText("7")).not.toBeInTheDocument();
  });

  it("polls every 30 seconds while the page is visible", async () => {
    vi.mocked(fetch).mockResolvedValue(
      Response.json({ stats: NEXT, updatedAt: UPDATED_AT }),
    );

    renderStats();
    expect(fetch).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(fetch).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("keeps the 30 second cadence across re-renders", async () => {
    vi.mocked(fetch).mockResolvedValue(
      Response.json({ stats: NEXT, updatedAt: UPDATED_AT }),
    );

    renderStats();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(20_000);
    });

    // A state update from a manual refresh re-renders the component.
    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Refresh metrics" }));
    });
    expect(fetch).toHaveBeenCalledTimes(1);

    // The poll still fires on the original 30s schedule, not 30s after the re-render.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("pauses polling while hidden and refreshes when visible again", async () => {
    vi.mocked(fetch).mockResolvedValue(
      Response.json({ stats: NEXT, updatedAt: UPDATED_AT }),
    );

    renderStats();

    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "hidden",
    });
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
    });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });
    expect(fetch).not.toHaveBeenCalled();

    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "visible",
    });
    await act(async () => {
      document.dispatchEvent(new Event("visibilitychange"));
    });
    expect(fetch).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(30_000);
    });
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it("shows an error and keeps the previous numbers when the request fails", async () => {
    vi.mocked(fetch).mockResolvedValue(new Response("", { status: 500 }));

    renderStats();

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Refresh metrics" }));
    });

    expect(screen.getByRole("alert")).toHaveTextContent("Something went wrong");
    expect(screen.getByText("11")).toBeInTheDocument();
    expect(screen.getByText("33")).toBeInTheDocument();
    expect(screen.getByText("22")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
  });
});
