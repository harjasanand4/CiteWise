"""Capture the README screenshots from a running server.

pip install playwright && playwright install chromium
citewise serve &                      # with whatever LLM you've configured in .env
python scripts/screenshots.py http://localhost:8000
"""

import asyncio
import sys
from pathlib import Path

from playwright.async_api import Page, async_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
ANSWER_Q = "How do I declare an incident?"
ABSTAIN_Q = "What's the dress code when I meet customers in person?"


async def ask(page: Page, question: str, n: int) -> None:
    await page.fill("#q", question)
    await page.keyboard.press("Enter")
    await page.wait_for_function(f"document.querySelectorAll('.turn-meta').length >= {n}", timeout=120_000)
    await page.wait_for_timeout(1200)  # let the highlighter animation finish


async def main(base: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=2)
        await page.goto(base, wait_until="networkidle")
        await page.wait_for_selector(".intro")
        await ask(page, ANSWER_Q, 1)
        await page.screenshot(path=OUT / "answer.png")
        await ask(page, ABSTAIN_Q, 2)
        await page.screenshot(path=OUT / "abstain.png")
        await browser.close()
    print(f"saved to {OUT}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"))
