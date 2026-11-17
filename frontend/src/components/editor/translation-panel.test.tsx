import { fireEvent, render, screen } from "@testing-library/react";
import { draft } from "@/lib/editor/document.test-fixture";
import { TranslationPanel } from "./translation-panel";

vi.mock("next-intl", () => ({
  useLocale: () => "en",
  useTranslations: () => (key: string) => key,
}));

it("applies an edited hook independently of incomplete subtitle translation", () => {
  const doc = { ...draft, hookTitle: "Original", localizedHookTitle: null };
  const update = vi.fn();
  render(<TranslationPanel doc={doc} endpoint="/editor" open onOpenChange={() => {}} update={update} />);
  fireEvent.change(screen.getByLabelText("captions.localizedHookTitle"), { target: { value: "Русский заголовок" } });
  const buttons = screen.getAllByRole("button", { name: "captions.translationApply" });
  fireEvent.click(buttons[0]);
  expect(update).toHaveBeenCalledTimes(1);
  const changed = update.mock.calls[0][0](doc);
  expect(changed.hookTitle).toBe("Original");
  expect(changed.localizedHookTitle).toBe("Русский заголовок");
  expect(changed.translation).toBe(doc.translation);
});
