'use client'

import { useTranslation } from 'react-i18next'

import {
  PreferencePage,
  PreferenceRow,
  PreferenceSection,
} from '@/components/preferences/PreferenceFields'
import { useForkPreferences } from '@/lib/forkPreferences'
import { Switch } from '@koharu/ui/components/switch'

export function FeaturesPreferences() {
  const { t } = useTranslation()
  const preferences = useForkPreferences((state) => state.preferences)
  const setPreference = useForkPreferences((state) => state.setPreference)

  return (
    <PreferencePage
      title={t('settings.features.title')}
      description={t('settings.features.description')}
    >
      <PreferenceSection title={t('settings.features.pages')}>
        <PreferenceRow
          title={t('settings.features.mergeDeleteOriginals')}
          description={t('settings.features.mergeDeleteOriginalsDescription')}
        >
          <div className='flex h-8 items-center justify-end'>
            <Switch
              aria-label={t('settings.features.mergeDeleteOriginals')}
              checked={preferences.mergeDeleteOriginals}
              onCheckedChange={(mergeDeleteOriginals) =>
                setPreference('mergeDeleteOriginals', mergeDeleteOriginals)
              }
            />
          </div>
        </PreferenceRow>
        <PreferenceRow
          title={t('settings.features.showPageNumbers')}
          description={t('settings.features.showPageNumbersDescription')}
        >
          <div className='flex h-8 items-center justify-end'>
            <Switch
              aria-label={t('settings.features.showPageNumbers')}
              checked={preferences.showPageNumbers}
              onCheckedChange={(showPageNumbers) =>
                setPreference('showPageNumbers', showPageNumbers)
              }
            />
          </div>
        </PreferenceRow>
      </PreferenceSection>
    </PreferencePage>
  )
}
