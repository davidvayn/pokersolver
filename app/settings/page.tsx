import { SettingsForm } from '@/components/settings/SettingsForm';

export default function SettingsPage() {
  return (
    <div className="mx-auto flex max-w-xl flex-col gap-6">
      <div>
        <h1 className="text-xl font-semibold">Settings</h1>
        <p className="text-sm text-muted">
          Configure solver details and the AI provider used for spot analysis.
          You can also open this from the gear icon in the header.
        </p>
      </div>
      <div className="rounded-lg border border-border bg-surface p-5">
        <SettingsForm />
      </div>
    </div>
  );
}
