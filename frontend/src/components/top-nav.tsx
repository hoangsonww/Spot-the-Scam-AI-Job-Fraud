"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import useSWR from "swr";
import { fetchReviewCount } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Menu, X } from "lucide-react";

export default function TopNav() {
  const pathname = usePathname() || "/";
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { data: reviewCount } = useSWR("review-count", fetchReviewCount, {
    refreshInterval: 60000,
    dedupingInterval: 100,
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
    { href: "/chat", label: "Chat" },
  ];

  return (
    <header className="sticky top-0 z-30 border-b border-slate-800/60 bg-black backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between px-4 py-3 sm:py-4 sm:px-8">
        <Link
          href="/"
          className="text-xs sm:text-sm font-semibold uppercase tracking-[0.15em] sm:tracking-[0.2em] text-slate-300"
        >
          Spot the Scam
        </Link>

        {/* Desktop Navigation */}
        <nav className="hidden sm:flex items-center gap-2 text-sm font-medium text-slate-300">
          {navLinks.map((link) => {
            const active = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`inline-flex items-center gap-2 rounded-full px-3 py-1 transition-colors ${
                  active ? "bg-slate-800 text-slate-100" : "hover:bg-slate-800/70 text-slate-300"
                }`}
              >
                <span>{link.label}</span>
                {"badge" in link && link.badge}
              </Link>
            );
          })}
        </nav>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setIsMenuOpen(!isMenuOpen)}
          className="sm:hidden p-2 text-slate-300 hover:text-slate-100 transition-colors"
          aria-label="Toggle menu"
        >
          {isMenuOpen ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </div>

      {/* Mobile Navigation */}
      {isMenuOpen && (
        <div className="sm:hidden border-t border-slate-800/60 bg-black">
          <nav className="flex flex-col px-4 py-2">
            {navLinks.map((link) => {
              const active = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setIsMenuOpen(false)}
                  className={`flex items-center justify-between gap-2 rounded-lg px-3 py-3 text-sm font-medium transition-colors ${
                    active ? "bg-slate-800 text-slate-100" : "text-slate-300 hover:bg-slate-800/70"
                  }`}
                >
                  <span>{link.label}</span>
                  {"badge" in link && link.badge}
                </Link>
              );
            })}
          </nav>
        </div>
      )}
    </header>
  );
}
