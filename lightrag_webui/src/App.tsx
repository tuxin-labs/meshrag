import { useState, useCallback, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import ThemeProvider from '@/components/ThemeProvider'
import ApiKeyAlert from '@/components/ApiKeyAlert'
import StatusIndicator from '@/components/status/StatusIndicator'
import { SiteInfo, webuiPrefix } from '@/lib/constants'
import { cn } from '@/lib/utils'
import { useBackendState, useAuthStore } from '@/stores/state'
import { useSettingsStore } from '@/stores/settings'
import { getAuthStatus } from '@/api/lightrag'
import SiteHeader from '@/features/SiteHeader'
import SideNav from '@/features/SideNav'
import { InvalidApiKeyError, RequireApiKeError } from '@/api/lightrag'

import GraphViewer from '@/features/GraphViewer'
import DocumentManager from '@/features/DocumentManager'
import RetrievalTesting from '@/features/RetrievalTesting'
import ApiSite from '@/features/ApiSite'
import ModelManager from '@/features/ModelManager'
import KnowledgeBaseManager from '@/features/KnowledgeBaseManager'
import ExternalKBManager from '@/features/ExternalKBManager'

import { Tabs, TabsContent } from '@/components/ui/Tabs'

function App() {
  const { t } = useTranslation()
  const message = useBackendState.use.message()
  const enableHealthCheck = useSettingsStore.use.enableHealthCheck()
  const currentTab = useSettingsStore.use.currentTab()
  const sidebarCollapsed = useSettingsStore.use.sidebarCollapsed()
  const [apiKeyAlertOpen, setApiKeyAlertOpen] = useState(false)
  const [initializing, setInitializing] = useState(true) // Add initializing state
  const versionCheckRef = useRef(false); // Prevent duplicate calls in Vite dev mode
  const healthCheckInitializedRef = useRef(false); // Prevent duplicate health checks in Vite dev mode

  const handleApiKeyAlertOpenChange = useCallback((open: boolean) => {
    setApiKeyAlertOpen(open)
    if (!open) {
      useBackendState.getState().clear()
    }
  }, [])

  // Track component mount status with useRef
  const isMountedRef = useRef(true);

  // Set up mount/unmount status tracking
  useEffect(() => {
    isMountedRef.current = true;

    // Handle page reload/unload
    const handleBeforeUnload = () => {
      isMountedRef.current = false;
    };

    window.addEventListener('beforeunload', handleBeforeUnload);

    return () => {
      isMountedRef.current = false;
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  // Health check - can be disabled
  useEffect(() => {
    // Health check function
    const performHealthCheck = async () => {
      try {
        // Only perform health check if component is still mounted
        if (isMountedRef.current) {
          await useBackendState.getState().check();
        }
      } catch (error) {
        console.error('Health check error:', error);
      }
    };

    // Set health check function in the store
    useBackendState.getState().setHealthCheckFunction(performHealthCheck);

    if (!enableHealthCheck || apiKeyAlertOpen) {
      useBackendState.getState().clearHealthCheckTimer();
      return;
    }

    // On first mount or when enableHealthCheck becomes true and apiKeyAlertOpen is false,
    // perform an immediate health check and start the timer
    if (!healthCheckInitializedRef.current) {
      healthCheckInitializedRef.current = true;
    }

    // Start/reset the health check timer using the store
    useBackendState.getState().resetHealthCheckTimer();

    // Component unmount cleanup
    return () => {
      useBackendState.getState().clearHealthCheckTimer();
    };
  }, [enableHealthCheck, apiKeyAlertOpen]);

  // Version check - independent and executed only once
  useEffect(() => {
    const checkVersion = async () => {
      // Prevent duplicate calls in Vite dev mode
      if (versionCheckRef.current) return;
      versionCheckRef.current = true;

      // Check if version info was already obtained in login page
      const versionCheckedFromLogin = sessionStorage.getItem('VERSION_CHECKED_FROM_LOGIN') === 'true';
      if (versionCheckedFromLogin) {
        setInitializing(false); // Skip initialization if already checked
        return;
      }

      try {
        setInitializing(true); // Start initialization

        // Get version info
        const token = localStorage.getItem('LIGHTRAG-API-TOKEN');
        const status = await getAuthStatus();

        // If auth is not configured and a new token is returned, use the new token
        if (!status.auth_configured && status.access_token) {
          useAuthStore.getState().login(
            status.access_token, // Use the new token
            true, // Guest mode
            status.core_version,
            status.api_version,
            status.webui_title || null,
            status.webui_description || null
          );
        } else if (token && (status.core_version || status.api_version || status.webui_title || status.webui_description)) {
          // Otherwise use the old token (if it exists)
          const isGuestMode = status.auth_mode === 'disabled' || useAuthStore.getState().isGuestMode;
          useAuthStore.getState().login(
            token,
            isGuestMode,
            status.core_version,
            status.api_version,
            status.webui_title || null,
            status.webui_description || null
          );
        }

        // Set flag to indicate version info has been checked
        sessionStorage.setItem('VERSION_CHECKED_FROM_LOGIN', 'true');
      } catch (error) {
        console.error('Failed to get version info:', error);
      } finally {
        // Ensure initializing is set to false even if there's an error
        setInitializing(false);
      }
    };

    // Execute version check
    checkVersion();
  }, []); // Empty dependency array ensures it only runs once on mount

  const handleTabChange = useCallback(
    (tab: string) => useSettingsStore.getState().setCurrentTab(tab as any),
    []
  )

  useEffect(() => {
    if (message) {
      if (message.includes(InvalidApiKeyError) || message.includes(RequireApiKeError)) {
        setApiKeyAlertOpen(true)
      }
    }
  }, [message])

  return (
    <ThemeProvider>
      {initializing ? (
      // Loading state mirroring the rail + header shell so hydration does not shift the layout
        <div className="flex h-screen w-screen overflow-hidden">
          <div
            aria-hidden="true"
            className={cn('bg-card border-r shrink-0', sidebarCollapsed ? 'w-14' : 'w-52')}
          />
          <div className="flex min-w-0 grow flex-col overflow-hidden">
            <header className="border-border/40 bg-background/95 supports-[backdrop-filter]:bg-background/60 flex h-10 w-full shrink-0 border-b px-4 backdrop-blur">
              <div className="flex min-w-0 flex-1 items-center">
                <a href={webuiPrefix} className="flex items-center gap-2">
                  <img src="logo.png" alt="" className="size-4" aria-hidden="true" />
                  <span className="font-bold md:inline-block">{SiteInfo.name}</span>
                </a>
              </div>
              <nav className="flex min-w-0 flex-1 items-center justify-end" />
            </header>

            <div className="flex flex-1 items-center justify-center">
              <div className="text-center">
                <div className="mx-auto mb-2 h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                <p className="text-sm text-muted-foreground">
                  {t('common.initializing', { defaultValue: 'Initializing...' })}
                </p>
              </div>
            </div>
          </div>
        </div>
      ) : (
      // Main content after initialization
        <main className="flex h-screen w-screen overflow-hidden">
          <Tabs
            defaultValue={currentTab}
            className="!m-0 flex grow !p-0 overflow-hidden"
            onValueChange={handleTabChange}
          >
            <SideNav />
            <div className="flex min-w-0 grow flex-col overflow-hidden">
              <SiteHeader />
              <div className="relative grow">
                <TabsContent value="documents" className="absolute top-0 right-0 bottom-0 left-0 overflow-auto">
                  <DocumentManager />
                </TabsContent>
                <TabsContent value="knowledge-graph" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
                  <GraphViewer />
                </TabsContent>
                <TabsContent value="retrieval" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
                  <RetrievalTesting />
                </TabsContent>
                <TabsContent value="knowledge-bases" className="absolute top-0 right-0 bottom-0 left-0 overflow-auto">
                  <KnowledgeBaseManager />
                </TabsContent>
                <TabsContent value="models" className="absolute top-0 right-0 bottom-0 left-0 overflow-auto">
                  <ModelManager />
                </TabsContent>
                <TabsContent value="external-kbs" className="absolute top-0 right-0 bottom-0 left-0 overflow-auto">
                  <ExternalKBManager />
                </TabsContent>
                <TabsContent value="api" className="absolute top-0 right-0 bottom-0 left-0 overflow-hidden">
                  <ApiSite />
                </TabsContent>
              </div>
            </div>
          </Tabs>
          {enableHealthCheck && <StatusIndicator />}
          <ApiKeyAlert open={apiKeyAlertOpen} onOpenChange={handleApiKeyAlertOpenChange} />
        </main>
      )}
    </ThemeProvider>
  )
}

export default App
