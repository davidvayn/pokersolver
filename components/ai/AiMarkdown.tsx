import React from "react";
import ReactMarkdown, { type Components } from "react-markdown";

interface AiMarkdownProps {
  content: string;
  className?: string;
}

const headingClass = "pt-1 text-sm font-semibold leading-snug text-fg";

const components: Components = {
  // AI content lives under the panel's h3, so its visual headings start at h4.
  h1: ({ children }) => <h4 className={headingClass}>{children}</h4>,
  h2: ({ children }) => <h4 className={headingClass}>{children}</h4>,
  h3: ({ children }) => <h4 className={headingClass}>{children}</h4>,
  h4: ({ children }) => <h4 className={headingClass}>{children}</h4>,
  h5: ({ children }) => <h4 className={headingClass}>{children}</h4>,
  h6: ({ children }) => <h4 className={headingClass}>{children}</h4>,
  p: ({ children }) => <p className="leading-relaxed">{children}</p>,
  ul: ({ children }) => (
    <ul className="list-disc space-y-1 pl-5 marker:text-muted">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="list-decimal space-y-1 pl-5 marker:text-muted">
      {children}
    </ol>
  ),
  li: ({ children }) => <li className="pl-0.5 leading-relaxed">{children}</li>,
  strong: ({ children }) => (
    <strong className="font-semibold text-fg">{children}</strong>
  ),
  em: ({ children }) => <em className="italic">{children}</em>,
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-accent pl-3 text-muted">
      {children}
    </blockquote>
  ),
  a: ({ children, href }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="text-accent underline underline-offset-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
    >
      {children}
    </a>
  ),
  pre: ({ children }) => (
    <pre className="overflow-x-auto rounded-md bg-bg p-3 font-mono text-xs leading-relaxed">
      {children}
    </pre>
  ),
  code: ({ children, className }) =>
    className ? (
      <code className={className}>{children}</code>
    ) : (
      <code className="rounded bg-bg px-1 py-0.5 font-mono text-[0.9em]">
        {children}
      </code>
    ),
  hr: () => <hr className="border-border" />,
};

export function AiMarkdown({ content, className = "" }: AiMarkdownProps) {
  return (
    <div className={`space-y-2 ${className}`}>
      <ReactMarkdown components={components}>{content}</ReactMarkdown>
    </div>
  );
}
