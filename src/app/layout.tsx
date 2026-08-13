import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Horizon — Long-Horizon Servicing Agent",
  description: "A customer service agent with a continuous context layer on MongoDB Atlas",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
