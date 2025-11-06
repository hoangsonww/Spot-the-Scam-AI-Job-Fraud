"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import useSWR from "swr";
import { fetchReviewCount } from "@/lib/api";
import { Badge } from "@/components/ui/badge";

export default function TopNav() {
  const pathname = usePathname() || "/";
  const { data: reviewCount } = useSWR("review-count", fetchReviewCount, {
    refreshInterval: 60000,
  });
  const count = reviewCount ?? 0;

  const navLinks = [
    { href: "/", label: "Score" },
    {
      href: "/review",
      label: "Review",
      badge: (
        <Badge
          variant="secondary"
          className={`text-xs font-semibold ${pathname === "/review" ? "" : "opacity-80"}`}
        >
          {count}
        </Badge>
      ),
    },
  ];

  return (
    <header className="sticky top-0 z-30 border-b border-slate-800/60 bg-slate-950/80 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-4 sm:px-8">
        <Link href="/" className="text-sm font-semibold uppercase tracking-[0.2em] text-slate-300">
          Spot the Scam
        </Link>
        <nav className="flex items-center gap-2 text-sm font-medium text-slate-300">
          {navLinks.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`inline-flex items-center gap-2 rounded-full px-3 py-1 transition-colors ${
                  active
                    ? "bg-slate-800 text-slate-100"
                    : "hover:bg-slate-800/70 text-slate-300"
                }`}
              >
                <span>{link.label}</span>
                {"badge" in link && link.badge}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
