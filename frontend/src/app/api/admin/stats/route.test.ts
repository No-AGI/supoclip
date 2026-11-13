import { GET } from "./route";
import { getPrismaClient } from "@/server/prisma";
import { getServerSession } from "@/server/session";

vi.mock("@/server/session", () => ({ getServerSession: vi.fn() }));
vi.mock("@/server/prisma", () => ({ getPrismaClient: vi.fn() }));

type CountArgs = { where?: { is_admin?: boolean; status?: string | { in: string[] } } } | undefined;

const USERS = 12;
const ADMINS = 3;
const TASKS = 40;
const COMPLETED = 25;
const PROCESSING = 5;
const QUEUED = 8;

describe("/api/admin/stats", () => {
  const user = { count: vi.fn() };
  const task = { count: vi.fn() };

  beforeEach(() => {
    vi.resetAllMocks();
    vi.mocked(getServerSession).mockResolvedValue({ user: { id: "admin-1", is_admin: true } } as never);
    vi.mocked(getPrismaClient).mockReturnValue({ user, task } as never);

    user.count.mockImplementation(async (args?: CountArgs) =>
      args?.where?.is_admin ? ADMINS : USERS,
    );
    task.count.mockImplementation(async (args?: CountArgs) => {
      const status = args?.where?.status;
      if (status === "completed") return COMPLETED;
      if (status === "processing") return PROCESSING;
      if (status && typeof status === "object" && "in" in status) return QUEUED;
      return TASKS;
    });
  });

  it("rejects anonymous callers", async () => {
    vi.mocked(getServerSession).mockResolvedValue(null as never);

    const response = await GET();

    expect(response.status).toBe(401);
    await expect(response.json()).resolves.toEqual({ error: "Unauthorized" });
    expect(user.count).not.toHaveBeenCalled();
    expect(task.count).not.toHaveBeenCalled();
  });

  it("rejects non-admin users", async () => {
    vi.mocked(getServerSession).mockResolvedValue({ user: { id: "user-2", is_admin: false } } as never);

    const response = await GET();

    expect(response.status).toBe(403);
    await expect(response.json()).resolves.toEqual({ error: "Forbidden" });
    expect(user.count).not.toHaveBeenCalled();
    expect(task.count).not.toHaveBeenCalled();
  });

  it("returns the admin stats", async () => {
    const response = await GET();

    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      stats: Record<string, number>;
      updatedAt: string;
    };
    expect(body.stats).toEqual({
      users: USERS,
      admins: ADMINS,
      tasks: TASKS,
      completed: COMPLETED,
      processing: PROCESSING,
      queued: QUEUED,
    });
    expect(Number.isNaN(Date.parse(body.updatedAt))).toBe(false);

    expect(task.count).toHaveBeenCalledWith({ where: { status: { in: ["queued", "pending"] } } });
    expect(task.count).toHaveBeenCalledWith({ where: { status: "processing" } });
  });

  it("returns 500 when the database call fails", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
    task.count.mockRejectedValue(new Error("database is down"));

    const response = await GET();

    expect(response.status).toBe(500);
    await expect(response.json()).resolves.toEqual({ error: "Internal server error" });
    expect(consoleError).toHaveBeenCalledWith("Failed to load admin stats:", expect.any(Error));

    consoleError.mockRestore();
  });
});
