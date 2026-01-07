import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Omni-Query — Multimodal RAG",
  description: "Multimodal RAG for Long-Document Intelligence. 222 PDFs · 4,055 QA pairs · Gemini 2.5 Flash",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full">
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
