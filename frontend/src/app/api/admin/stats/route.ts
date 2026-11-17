import { NextResponse } from "next/server";

import { getPrismaClient } from "@/server/prisma";
import { getServerSession } from "@/server/session";

export const dynamic = "force-dynamic";

async function requireAdmin() {
  const session = await getServerSession();
  if (!session?.user?.id) {
    return { error: NextResponse.json({ error: "Unauthorized" }, { status: 401 }) };
  }

  const isAdmin = Boolean((session.user as { is_admin?: boolean }).is_admin);
  if (!isAdmin) {
    return { error: NextResponse.json({ error: "Forbidden" }, { status: 403 }) };
  }

  return { session };
}

export async function GET() {
  const adminCheck = await requireAdmin();
  if (adminCheck.error) {
    return adminCheck.error;
  }

  try {
    const prisma = getPrismaClient();

    // "queued" intentionally covers both waiting states (queued + pending);
    // "processing" is counted alone so the two numbers never overlap.
    const [users, admins, tasks, completed, processing, queued] = await Promise.all([
      prisma.user.count(),
      prisma.user.count({ where: { is_admin: true } }),
      prisma.task.count(),
      prisma.task.count({ where: { status: "completed" } }),
      prisma.task.count({ where: { status: "processing" } }),
      prisma.task.count({ where: { status: { in: ["queued", "pending"] } } }),
    ]);

    return NextResponse.json(
      {
        stats: { users, admins, tasks, completed, processing, queued },
        updatedAt: new Date().toISOString(),
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch (error) {
    console.error("Failed to load admin stats:", error);
    return NextResponse.json({ error: "Internal server error" }, { status: 500 });
  }
}
