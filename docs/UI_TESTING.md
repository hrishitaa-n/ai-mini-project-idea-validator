# UI testing checklist

Start the application with `python src/server.py`, then open `http://127.0.0.1:8000`.

- [ ] Check the page at 360 px and 1440 px in both light and dark system themes.
- [ ] Tab through the form: every control has a visible focus outline; Enter or Space changes chips and dataset choice; Ctrl/Cmd+Enter submits.
- [ ] Submit each sample idea and confirm score cards, prior work, products, flags, and the report area render:
  - `A spam email classifier using NLP`
  - `A lost-and-found matching system for a college campus`
  - `Generate college timetables with a genetic algorithm`
- [ ] Check empty input and an idea longer than 1500 characters. The first should show a friendly validation message; the second should validate and render normally.
- [ ] Stop the server and refresh: the form should show a network/settings error rather than breaking.
- [ ] With no `GEMINI_API_KEY`, submit an idea and confirm the one-line report-disabled note. With a deliberately failing report request, confirm the inline error and Try again button preserve the other results.
- [ ] Temporarily return a match with title/abstract `<img src=x onerror=alert(1)>` and URL `javascript:alert(1)`. Confirm the text is literal, no alert runs, and no source link is rendered. Remove the temporary response before committing.

## Verification notes

The API was exercised on 2026-11-18 with all three sample ideas. Prior work and product results returned successfully. The timetable query returned one weak product match (Khan Academy, similarity 0.331), rather than the expected empty product list; this originates from the current product-retrieval threshold and was not changed because `src/products.py` is protected by project rules.

Visual browser, console, keyboard-only, and theme checks could not be automated in this environment: local-browser automation was blocked by URL-policy enforcement. The frontend uses `textContent` and DOM text nodes for all API content, and accepts only `http`/`https` URLs before creating a link; the XSS case was reviewed statically but still needs the manual check above.
