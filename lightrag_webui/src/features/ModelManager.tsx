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
  createModelProfile,
  deleteModelProfile,
  listModelProfiles,
  supportedModelBindings,
  testModelProfile,
  updateModelProfile
} from '@/api/lightrag'
import type { ConnectionTestResult, ModelBinding, ModelKind, ModelProfile, ModelProfileInput } from '@/api/lightrag'
import { useSettingsStore } from '@/stores/settings'
import { errorMessage } from '@/lib/utils'
import { LoaderIcon, PencilIcon, PlugZapIcon, PlusIcon, RefreshCwIcon, Trash2Icon } from 'lucide-react'

const emptyForm = (): ModelProfileInput => ({
  name: '',
  kind: 'llm',
  binding: 'openai',
  model: '',
  host: '',
  api_key: '',
  embedding_dim: 1024,
  enabled: true
})

const toForm = (profile: ModelProfile): ModelProfileInput => ({
  name: profile.name,
  kind: profile.kind,
  binding: profile.binding,
  model: profile.model,
  host: profile.host,
  // Empty means "keep the stored key" for an existing profile.
  api_key: '',
  embedding_dim: profile.embedding_dim ?? 1024,
  enabled: profile.enabled
})

export default function ModelManager() {
  const { t } = useTranslation()
  const profiles = useSettingsStore.use.availableModelProfiles()
  const setProfiles = useSettingsStore.use.setAvailableModelProfiles()
  const [isLoading, setIsLoading] = useState(false)
  const [editing, setEditing] = useState<ModelProfile | null>(null)
  const [isFormOpen, setIsFormOpen] = useState(false)
  const [form, setForm] = useState<ModelProfileInput>(emptyForm())
  const [isSaving, setIsSaving] = useState(false)
  const [testingId, setTestingId] = useState<string | null>(null)
  const [results, setResults] = useState<Record<string, ConnectionTestResult>>({})
  const [pendingDelete, setPendingDelete] = useState<ModelProfile | null>(null)

  const refresh = useCallback(async () => {
    setIsLoading(true)
    try {
      setProfiles(await listModelProfiles())
    } catch (error) {
      toast.error(errorMessage(error))
    } finally {
      setIsLoading(false)
    }
  }, [setProfiles])

  useEffect(() => {
    refresh()
  }, [refresh])

  const openCreate = () => {
    setEditing(null)
    setForm(emptyForm())
    setIsFormOpen(true)
  }

  const openEdit = (profile: ModelProfile) => {
    setEditing(profile)
    setForm(toForm(profile))
    setIsFormOpen(true)
  }

  const handleSave = async () => {
    if (!form.name.trim() || !form.model.trim() || !form.host.trim()) {
      toast.error(t('modelManager.fieldsRequired', { defaultValue: 'Name, model and host are required' }))
      return
    }

    setIsSaving(true)
    try {
      // Omit an untouched key so the stored secret survives the edit.
      const payload: ModelProfileInput = { ...form, host: form.host.trim() }
      if (editing && !payload.api_key) {
        delete payload.api_key
      }
      if (payload.kind !== 'embedding') {
        delete payload.embedding_dim
      }

      if (editing) {
        await updateModelProfile(editing.id, payload)
      } else {
        await createModelProfile(payload)
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

  const handleTest = async (profile: ModelProfile) => {
    setTestingId(profile.id)
    try {
      const result = await testModelProfile(profile.id)
      setResults((prev) => ({ ...prev, [profile.id]: result }))
      if (!result.reachable || !result.model_available) {
        toast.warning(`${profile.name}: ${result.message}`)
      }
    } catch (error) {
      setResults((prev) => ({
        ...prev,
        [profile.id]: {
          id: profile.id,
          name: profile.name,
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
      await deleteModelProfile(pendingDelete.id)
      toast.success(t('modelManager.deleted', { defaultValue: 'Model profile deleted' }))
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
              <CardTitle>{t('modelManager.title', { defaultValue: 'Model Management' })}</CardTitle>
              <CardDescription>
                {t('modelManager.description', {
                  defaultValue: 'Endpoints are stored on the server; API keys are never sent to the browser.'
                })}
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={refresh} disabled={isLoading}>
                {isLoading ? <LoaderIcon className="animate-spin" /> : <RefreshCwIcon />}
                {t('common.refresh', { defaultValue: 'Refresh' })}
              </Button>
              <Button variant="default" size="sm" onClick={openCreate}>
                <PlusIcon />
                {t('modelManager.create', { defaultValue: 'Add model' })}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="m-0 grow overflow-auto p-0 text-xs">
          <Table className="w-full">
            <TableHeader className="sticky top-0 bg-background z-10">
              <TableRow className="border-b">
                <TableHead>{t('modelManager.columns.name', { defaultValue: 'Name' })}</TableHead>
                <TableHead>{t('modelManager.columns.kind', { defaultValue: 'Kind' })}</TableHead>
                <TableHead>{t('modelManager.columns.binding', { defaultValue: 'Binding' })}</TableHead>
                <TableHead>{t('modelManager.columns.model', { defaultValue: 'Model' })}</TableHead>
                <TableHead>{t('modelManager.columns.host', { defaultValue: 'Host' })}</TableHead>
                <TableHead>{t('modelManager.columns.apiKey', { defaultValue: 'Key' })}</TableHead>
                <TableHead className="w-24">{t('modelManager.columns.actions', { defaultValue: 'Actions' })}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody className="text-sm">
              {profiles.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7} className="text-muted-foreground text-center py-8">
                    {t('modelManager.empty', { defaultValue: 'No model profiles yet.' })}
                  </TableCell>
                </TableRow>
              )}
              {profiles.map((profile) => {
                const result = results[profile.id]
                return (
                  <Fragment key={profile.id}>
                    <TableRow>
                    <TableCell className="font-medium">{profile.name}</TableCell>
                    <TableCell>
                      <Badge variant={profile.kind === 'llm' ? 'default' : 'secondary'}>{profile.kind}</Badge>
                    </TableCell>
                    <TableCell>{profile.binding}</TableCell>
                    <TableCell className="max-w-40 truncate" title={profile.model}>{profile.model}</TableCell>
                    <TableCell className="max-w-56 truncate" title={profile.host}>{profile.host}</TableCell>
                    <TableCell>
                      {profile.has_api_key ? `••••${profile.api_key_tail}` : (
                        <span className="text-muted-foreground">{t('modelManager.noKey', { defaultValue: 'none' })}</span>
                      )}
                    </TableCell>
                    <TableCell className="text-center">
                      <div className="flex items-center justify-center gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-7"
                          tooltip={t('modelManager.test', { defaultValue: 'Test connection' })}
                          onClick={() => handleTest(profile)}
                          disabled={testingId === profile.id}
                        >
                          {testingId === profile.id
                            ? <LoaderIcon className="size-3.5 animate-spin" />
                            : <PlugZapIcon className="size-3.5" />}
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-7"
                          tooltip={t('common.edit', { defaultValue: 'Edit' })}
                          onClick={() => openEdit(profile)}
                        >
                          <PencilIcon className="size-3.5" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-7"
                          tooltip={t('common.delete')}
                          onClick={() => setPendingDelete(profile)}
                        >
                          <Trash2Icon className="size-3.5" />
                        </Button>
                      </div>
                    </TableCell>
                    </TableRow>
                    {result && (
                      <TableRow className="bg-muted/40">
                        <TableCell colSpan={7} className="py-1">
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
                ? t('modelManager.editTitle', { defaultValue: 'Edit Model Profile' })
                : t('modelManager.createTitle', { defaultValue: 'Add Model Profile' })}
            </DialogTitle>
            <DialogDescription>
              {t('modelManager.formDescription', {
                defaultValue: 'Embedding profiles also need the vector dimension their service returns.'
              })}
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-col gap-2">
            <Input
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder={t('modelManager.namePlaceholder', { defaultValue: 'Display name' })}
              disabled={isSaving}
            />
            <div className="flex gap-2">
              <Select
                value={form.kind}
                onValueChange={(value) => setForm({ ...form, kind: value as ModelKind })}
              >
                <SelectTrigger className="h-9 flex-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {(['llm', 'embedding', 'rerank'] as ModelKind[]).map((kind) => (
                      <SelectItem key={kind} value={kind}>{kind}</SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
              <Select
                value={form.binding}
                onValueChange={(value) => setForm({ ...form, binding: value as ModelBinding })}
              >
                <SelectTrigger className="h-9 flex-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectGroup>
                    {supportedModelBindings.map((binding) => (
                      <SelectItem key={binding} value={binding}>{binding}</SelectItem>
                    ))}
                  </SelectGroup>
                </SelectContent>
              </Select>
            </div>
            <Input
              value={form.model}
              onChange={(e) => setForm({ ...form, model: e.target.value })}
              placeholder={t('modelManager.modelPlaceholder', { defaultValue: 'Model name' })}
              disabled={isSaving}
            />
            <Input
              value={form.host}
              onChange={(e) => setForm({ ...form, host: e.target.value })}
              placeholder={t('modelManager.hostPlaceholder', { defaultValue: 'Base url, e.g. http://host:11434' })}
              disabled={isSaving}
            />
            <Input
              type="password"
              value={form.api_key ?? ''}
              onChange={(e) => setForm({ ...form, api_key: e.target.value })}
              placeholder={editing
                ? t('modelManager.keyKeepPlaceholder', { defaultValue: 'API key — leave empty to keep current' })
                : t('modelManager.keyPlaceholder', { defaultValue: 'API key (optional)' })}
              disabled={isSaving}
            />
            {form.kind === 'embedding' && (
              <Input
                type="number"
                value={form.embedding_dim ?? ''}
                onChange={(e) => setForm({ ...form, embedding_dim: parseInt(e.target.value) || 0 })}
                placeholder={t('modelManager.dimPlaceholder', { defaultValue: 'Embedding dimension' })}
                disabled={isSaving}
              />
            )}
            <label className="flex cursor-pointer items-center gap-2 text-sm">
              <Checkbox
                checked={form.enabled}
                onCheckedChange={(checked) => setForm({ ...form, enabled: checked === true })}
              />
              <span>{t('modelManager.enabled', { defaultValue: 'Enabled' })}</span>
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
            <AlertDialogTitle>{t('modelManager.deleteTitle', { defaultValue: 'Delete model profile' })}</AlertDialogTitle>
            <AlertDialogDescription>
              {t('modelManager.deleteDescription', {
                defaultValue: 'Delete "{{name}}"? Queries selecting it will stop working.',
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
