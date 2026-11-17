import { POST } from "./route";
import { getServerSession } from "@/server/session";
import { fetchBackend } from "@/server/backend-api";

vi.mock("@/server/session", () => ({ getServerSession: vi.fn() }));
vi.mock("@/server/backend-api", () => ({ fetchBackend: vi.fn(), createTextProxyResponse: (response: Response) => response }));

const context = { params: Promise.resolve({ path: ["connections"] }) };

beforeEach(() => {
  vi.resetAllMocks();
  vi.mocked(getServerSession).mockResolvedValue({ user: { id: "admin", is_admin: true } } as never);
  vi.mocked(fetchBackend).mockResolvedValue(new Response("{}") as never);
});

it("rejects cross-site requests and non-JSON bodies before forwarding", async () => {
  const url = "http://localhost/api/admin/llm/connections";
  expect((await POST(new Request(url, { method: "POST", headers: { origin: "https://evil.example", "content-type": "application/json" }, body: "{}" }), context)).status).toBe(403);
  expect((await POST(new Request(url, { method: "POST", headers: { "content-type": "text/plain" }, body: "{}" }), context)).status).toBe(415);
  expect(fetchBackend).not.toHaveBeenCalled();
});

it("forwards same-origin JSON for an administrator", async () => {
  const url = "http://localhost/api/admin/llm/connections";
  const response = await POST(new Request(url, { method: "POST", headers: { origin: "http://localhost", "content-type": "application/json" }, body: "{}" }), context);
  expect(response.status).toBe(200);
  expect(fetchBackend).toHaveBeenCalledOnce();
});
