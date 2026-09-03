import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { AiMarkdown } from "./AiMarkdown";

describe("AiMarkdown", () => {
  it("renders common AI response Markdown as semantic content", () => {
    const html = renderToStaticMarkup(
      createElement(AiMarkdown, {
        content:
          "### Recommendation\n\n- Bet **33% pot** with `AK`.\n- Check marginal hands.",
      }),
    );

    expect(html).toContain("<h4");
    expect(html).toContain("<ul");
    expect(html).toContain("<strong");
    expect(html).toContain("<code");
    expect(html).not.toContain("### Recommendation");
  });
});
