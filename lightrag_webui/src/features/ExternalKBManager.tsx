import { Fragment, useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import Checkbox from '@/components/ui/Checkbox'
import Input from '@/components/ui/Input'
import Badge from '@/components/ui/Badge'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/Table'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle
} from '@/components/ui/AlertDialog'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/Select'
import {
  createExternalKB,
  deleteExternalKB,
  listExternalKBs,
  testExternalKB,
  updateExternalKB
} from '@/api/lightrag'
import type {
  ConnectionTestResult,
  ExternalKBInput,
  ExternalKBRecord,
  ExternalKBType
} from '@/api/lightrag'
import { useSettingsStore } from '@/stores/settings'
import { errorMessage } from '@/lib/utils'
import { LoaderIcon, PencilIcon, PlugZapIcon, PlusIcon, RefreshCwIcon, Trash2Icon, EllipsisIcon } from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator
} from '@/components/ui/DropdownMenu'

const emptyForm = (): ExternalKBInput => ({
  name: '',
  type: 'retrieval',
  url: '',
  api_key: '',
  top_k: 5,
  enabled: true
})

const toForm = (kb: ExternalKBRecord): ExternalKBInput => ({
  name: kb.name,
  type: kb.type,
  url: kb.url,
  api_key: '',
  top_k: kb.top_k,
  enabled: kb.enabled
})

export default function ExternalKBManager() {
  const { t } = useTranslation()
  const kbs = useSettingsStore.use.availableExternalKBs()
  const setKbs = useSettingsStore.use.setAvailableExternalKBs()
  const [isLoading, setIsLoading] = useState(false)
  const [editing, setEditing] = useState<ExternalKBRecord | null>(null)
  const [isFormOpen, setIsFormOpen] = useState(false)
  const [form, setForm] = useState<ExternalKBInput>(emptyForm())
  const [isSaving, setIsSaving] = useState(false)
  const [testingId, setTestingId] = useState<string | null>(null)
  const [results, setResults] = useState<Record<string, ConnectionTestResult>>({})
  const [pendingDelete, setPendingDelete] = useState<ExternalKBRecord | null>(null)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    try {
      setKbs(await listExternalKBs())
    } catch (error) {
      toast.error(errorMessage(error))
    } finally {
      setIsLoading(false)
    }
  }, [setKbs])

  useEffect(() => {
    refresh()
  }, [refresh])

  const handleSave = async () => {
    if (!form.name.trim() || !form.url.trim()) {
      toast.error(t('externalKBManager.fieldsRequired', { defaultValue: 'Name and url are required' }))
      return
    }

    setIsSaving(true)
    try {
      const payload: ExternalKBInput = { ...form, name: form.name.trim(), url: form.url.trim() }
      // An empty key field on an existing entry means "keep what is stored".
      if (editing && !payload.api_key) {
        delete payload.api_key
      }
      if (editing) {
        await updateExternalKB(editing.id, payload)
      } else {
        await createExternalKB(payload)
      }
      toast.success(t('common.saveSuccess', { defaultValue: 'Saved' }))
      setIsFormOpen(false)
      setEditing(null)
      await refresh()
    } catch (error) {
      toast.error(errorMessage(error))
    } finally {
      setIsSaving(false)
    }
  }

  const handleTest = async (kb: ExternalKBRecord) => {
    setTestingId(kb.id)
    try {
      const result = await testExternalKB(kb.id)
      setResults((prev) => ({ ...prev, [kb.id]: result }))
      if (!result.reachable || !result.model_available) {
        toast.warning(`${kb.name}: ${result.message}`)
      }
    } catch (error) {
      setResults((prev) => ({
        ...prev,
        [kb.id]: {
          id: kb.id,
          name: kb.name,
          reachable: false,
          model_available: false,
          message: errorMessage(error),
          latency_ms: 0
        }
      }))
    } finally {
      setTestingId(null)
    }
  }

  const handleDelete = async () => {
    if (!pendingDelete) return
    try {
      await deleteExternalKB(pendingDelete.id)
      toast.success(t('externalKBManager.deleted', { defaultValue: 'External knowledge base deleted' }))
      setPendingDelete(null)
      await refresh()
    } catch (error) {
      toast.error(errorMessage(error))
    }
  }

  return (
    <div className="flex size-full gap-2 px-2 pb-12 overflow-hidden">
      <Card className="flex grow flex-col overflow-hidden">
        <CardHeader className="px-4 pt-4 pb-2">
          <div className="flex items-center justify-between gap-2">
            <div>
              <CardTitle>{t('externalKBManager.title', { defaultValue: 'External Knowledge Bases' })}</CardTitle>
              <CardDescription>
                {t('externalKBManager.description', {
                  defaultValue: 'Registered services can be selected per query alongside internal knowledge bases.'
                })}
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={refresh} disabled={isLoading}>
                {isLoading ? <LoaderIcon className="animate-spin" /> : <RefreshCwIcon />}
                {t('common.refresh', { defaultValue: 'Refresh' })}
              </Button>
              <Button
                variant="default"
                size="sm"
                onClick={() => {
                  setEditing(null)
                  setForm(emptyForm())
                  setIsFormOpen(true)
                }}
              >
                <PlusIcon />
                {t('externalKBManager.create', { defaultValue: 'Add external KB' })}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="m-0 grow overflow-auto p-0 text-xs">
          <Table className="w-full">
            <TableHeader className="sticky top-0 bg-background z-10">
              <TableRow className="border-b">
                <TableHead>{t('externalKBManager.columns.name', { defaultValue: 'Name' })}</TableHead>
                <TableHead>{t('externalKBManager.columns.type', { defaultValue: 'Type' })}</TableHead>
                <TableHead>{t('externalKBManager.columns.url', { defaultValue: 'Url' })}</TableHead>
                <TableHead>{t('externalKBManager.columns.topK', { defaultValue: 'Top K' })}</TableHead>
                <TableHead>{t('externalKBManager.columns.apiKey', { defaultValue: 'Key' })}</TableHead>
                <TableHead className="w-24">{t('externalKBManager.columns.actions', { defaultValue: 'Actions' })}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody className="text-sm">
              {kbs.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} className="text-muted-foreground text-center py-8">
                    {t('externalKBManager.empty', { defaultValue: 'No external knowledge bases registered.' })}
                  </TableCell>
                </TableRow>
              )}
              {kbs.map((kb) => {
                const result = results[kb.id]
                return (
                  <Fragment key={kb.id}>
                    <TableRow>
                      <TableCell className="font-medium">{kb.name}</TableCell>
                      <TableCell>
                        <Badge variant={kb.type === 'rag' ? 'default' : 'secondary'}>{kb.type}</Badge>
                      </TableCell>
                      <TableCell className="max-w-72 truncate" title={kb.url}>{kb.url}</TableCell>
                      <TableCell>{kb.type === 'retrieval' ? kb.top_k : '-'}</TableCell>
                      <TableCell>
                        {kb.has_api_key ? `••••${kb.api_key_tail}` : (
                          <span className="text-muted-foreground">{t('externalKBManager.noKey', { defaultValue: 'none' })}</span>
                        )}
                      </TableCell>
                      <TableCell className="text-center">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="size-7"
                              tooltip={t('common.actions', { defaultValue: 'Actions' })}
                              disabled={testingId === kb.id}
                            >
                              {testingId === kb.id
                                ? <LoaderIcon className="size-3.5 animate-spin" />
                                : <EllipsisIcon className="size-4" />}
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem onClick={() => handleTest(kb)}>
                              <PlugZapIcon />
                              {t('externalKBManager.test', { defaultValue: 'Test connection' })}
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={() => {
                                setEditing(kb)
                                setForm(toForm(kb))
                                setIsFormOpen(true)
                              }}
                            >
                              <PencilIcon />
                              {t('common.edit', { defaultValue: 'Edit' })}
                            </DropdownMenuItem>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem destructive onClick={() => setPendingDelete(kb)}>
                              <Trash2Icon />
                              {t('common.delete')}
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </TableCell>
                    </TableRow>
                    {result && (
                      <TableRow className="bg-muted/40">
                        <TableCell colSpan={6} className="py-1">
                          <span className={result.reachable && result.model_available ? 'text-green-600' : 'text-amber-600'}>
                            {result.message}
                          </span>
                          <span className="text-muted-foreground ml-2">{result.latency_ms} ms</span>
                          {result.detail && (
                            <span className="text-muted-foreground ml-2">{JSON.stringify(result.detail)}</span>
                          )}
                        </TableCell>
                      </TableRow>
                    )}
                  </Fragment>
                )
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog open={isFormOpen} onOpenChange={(open) => !isSaving && setIsFormOpen(open)}>
        <DialogContent className="sm:max-w-md" onCloseAutoFocus={(e) => e.preventDefault()}>
          <DialogHeader>
            <DialogTitle>
              {editing
                ? t('externalKBManager.editTitle', { defaultValue: 'Edit External Knowledge Base' })
                : t('externalKBManager.createTitle', { defaultValue: 'Add External Knowledge Base' })}
            </DialogTitle>
            <DialogDescription>
              {t('externalKBManager.formDescription', {
                defaultValue: 'retrieval returns text chunks for the local LLM; rag returns a finished answer.'
              })}
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-col gap-2">
            <Input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder={t('externalKBManager.namePlaceholder', { defaultValue: 'Display name' })}
              disabled={isSaving}
            />
            <div className="flex gap-2">
              <Select
                value={form.type}
                onValueChange={(value) => setForm({ ...form, type: value as ExternalKBType })}
              >
                <SelectTrigger className="h-9 flex-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    <SelectItem value="retrieval">
                      {t('retrievePanel.querySettings.externalKbRetrieval', { defaultValue: 'Retrieval' })}
                    </SelectItem>
                    <SelectItem value="rag">
                      {t('retrievePanel.querySettings.externalKbRag', { defaultValue: 'RAG service' })}
                    </SelectItem>
                  </SelectGroup>
                </SelectContent>
              </Select>
              {form.type === 'retrieval' && (
                <Input
                  type="number"
                  min={1}
                  max={50}
                  value={form.top_k}
                  onChange={(e) =>
                    setForm({ ...form, top_k: Math.min(50, Math.max(1, parseInt(e.target.value) || 1)) })
                  }
                  className="h-9 w-24"
                  disabled={isSaving}
                />
              )}
            </div>
            <Input
              value={form.url}
              onChange={(e) => setForm({ ...form, url: e.target.value })}
              placeholder={t('externalKBManager.urlPlaceholder', { defaultValue: 'https://service.example.com/api' })}
              disabled={isSaving}
            />
            <Input
              type="password"
              value={form.api_key ?? ''}
              onChange={(e) => setForm({ ...form, api_key: e.target.value })}
              placeholder={editing
                ? t('externalKBManager.keyKeepPlaceholder', { defaultValue: 'API key — leave empty to keep current' })
                : t('externalKBManager.keyPlaceholder', { defaultValue: 'API key (optional)' })}
              disabled={isSaving}
            />
            <label className="flex cursor-pointer items-center gap-2 text-sm">
              <Checkbox
                checked={form.enabled}
                onCheckedChange={(checked) => setForm({ ...form, enabled: checked === true })}
              />
              <span>{t('externalKBManager.enabled', { defaultValue: 'Enabled' })}</span>
            </label>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setIsFormOpen(false)} disabled={isSaving}>
              {t('common.cancel')}
            </Button>
            <Button onClick={handleSave} disabled={isSaving}>
              {isSaving ? t('common.saving') : t('common.save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={!!pendingDelete} onOpenChange={(open) => !open && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('externalKBManager.deleteTitle', { defaultValue: 'Delete external knowledge base' })}</AlertDialogTitle>
            <AlertDialogDescription>
              {t('externalKBManager.deleteDescription', {
                defaultValue: 'Delete "{{name}}" from the system?',
                name: pendingDelete?.name ?? ''
              })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete}>{t('common.delete')}</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
