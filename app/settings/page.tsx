import { SettingsForm } from '@/components/settings/SettingsForm';
import { PageHeading } from '@/components/design/DesignShell';

export default function SettingsPage() {
  return (
    <div className="settings-study">
      <PageHeading title="Settings" meta="Your workspace, your preferences." />
      <div className="settings-layout">
        <SettingsForm page />
        <aside className="settings-help">
          <div>
            <h2>Private by default</h2>
            <p>
              Practice history stays on this device. API keys stay in this
              browser.
            </p>
          </div>
          <div>
            <h2>Connect your analysis</h2>
            <p>
              Choose a provider and model. Your key is sent only with analysis
              requests.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}
