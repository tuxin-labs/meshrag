import { useCallback, useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import Badge from '@/components/ui/Badge'
import Input from '@/components/ui/Input'
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
import type { KnowledgeBaseStats } from '@/api/lightrag'
import { useKnowledgeBase } from '@/hooks/useKnowledgeBase'
import { useSettingsStore } from '@/stores/settings'
import { errorMessage } from '@/lib/utils'
import {
  CheckIcon,
  EllipsisIcon,
  LoaderIcon,
  PlusIcon,
  RefreshCwIcon,
  Trash2Icon,
  TriangleAlertIcon
} from 'lucide-react'
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem
} from '@/components/ui/DropdownMenu'

type KbRow = {
  id: string
  total: number | null
  entities: number | null
  processed: number
  processing: number
  pending: number
  failed: number
}

const countOf = (counts: Record<string, number>, ...keys: string[]) => {
  for (const key of keys) {
    const value = counts[key] ?? counts[key.toUpperCase()]
    if (typeof value === 'number') {
      return value
    }
  }
  return 0
}

/** A base the stats endpoint did not report on, e.g. an older server without that route. */
const emptyRow = (id: string): KbRow => ({
  id,
  total: null,
  entities: null,
  processed: 0,
  processing: 0,
  pending: 0,
  failed: 0
})

const toRow = (stats: KnowledgeBaseStats): KbRow => ({
  id: stats.kb_id,
  total: stats.documents,
  entities: stats.entities,
  processed: countOf(stats.documents_by_status, 'processed'),
  processing: countOf(stats.documents_by_status, 'processing'),
  pending: countOf(stats.documents_by_status, 'pending'),
  failed: countOf(stats.documents_by_status, 'failed')
})

export default function KnowledgeBaseManager() {
  const { t } = useTranslation()
  const selectedKbId = useSettingsStore.use.selectedKbId()
  const availableKbIds = useSettingsStore.use.availableKbIds()
  const defaultKbId = useSettingsStore.use.defaultKbId()
  const {
    refreshKnowledgeBases,
    refreshKnowledgeBaseStats,
    selectKnowledgeBase,
    createAndSelectKnowledgeBase,
    removeKnowledgeBase
  } = useKnowledgeBase()

  const [rows, setRows] = useState<KbRow[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [isCreating, setIsCreating] = useState(false)
  const [newKbId, setNewKbId] = useState('')
  const [pendingDelete, setPendingDelete] = useState<KbRow | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState('')
  const [isDeleting, setIsDeleting] = useState(false)

  const load = useCallback(async () => {
    setIsLoading(true)
    try {
      const [kbIds, statsById] = await Promise.all([
        refreshKnowledgeBases(),
        refreshKnowledgeBaseStats()
      ])
      setRows(kbIds.map((id) => (statsById[id] ? toRow(statsById[id]) : emptyRow(id))))
    } catch (error) {
      toast.error(errorMessage(error))
    } finally {
      setIsLoading(false)
    }
  }, [refreshKnowledgeBases, refreshKnowledgeBaseStats])

  useEffect(() => {
    load()
  }, [load])

  const handleCreate = async () => {
    const kbId = newKbId.trim()
    if (!kbId) {
      toast.error(t('header.kbCreateRequired', { defaultValue: 'Knowledge base ID is required' }))
      return
    }
    if (availableKbIds.includes(kbId)) {
      toast.error(t('kbManager.duplicateId', { defaultValue: 'That knowledge base already exists' }))
      return
    }

    setIsCreating(true)
    try {
      await createAndSelectKnowledgeBase(kbId)
      toast.success(t('header.kbCreateSuccess', { defaultValue: 'Knowledge base created' }))
      setNewKbId('')
      setIsCreateOpen(false)
      await load()
    } catch (error) {
      toast.error(errorMessage(error))
    } finally {
      setIsCreating(false)
    }
  }

  const handleDelete = async () => {
    if (!pendingDelete) return
    setIsDeleting(true)
    try {
      const result = await removeKnowledgeBase(pendingDelete.id)
      if (result.status === 'partial_success') {
        toast.warning(t('header.kbDeletePartial', {
          defaultValue: 'Knowledge base deleted, but some storages failed: {{storages}}',
          storages: result.storage_results?.failed?.join(', ') || ''
        }))
      } else {
        toast.success(t('header.kbDeleteSuccess', { defaultValue: 'Knowledge base deleted' }))
      }
      setPendingDelete(null)
      setDeleteConfirm('')
      await load()
    } catch (error) {
      toast.error(errorMessage(error))
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div className="flex size-full gap-2 px-2 pb-12 overflow-hidden">
      <Card className="flex grow flex-col overflow-hidden">
        <CardHeader className="px-4 pt-4 pb-2">
          <div className="flex items-center justify-between gap-2">
            <div>
              <CardTitle>{t('kbManager.title', { defaultValue: 'Knowledge Bases' })}</CardTitle>
              <CardDescription>
                {t('kbManager.description', {
                  defaultValue: 'Each base keeps its own graph, vectors and documents.'
                })}
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={load} disabled={isLoading}>
                {isLoading ? <LoaderIcon className="animate-spin" /> : <RefreshCwIcon />}
                {t('common.refresh', { defaultValue: 'Refresh' })}
              </Button>
              <Button variant="default" size="sm" onClick={() => setIsCreateOpen(true)}>
                <PlusIcon />
                {t('kbManager.create', { defaultValue: 'New knowledge base' })}
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent className="m-0 grow overflow-auto p-0">
          <Table className="w-full">
            <TableHeader className="sticky top-0 bg-background z-10">
              <TableRow className="border-b">
                <TableHead>{t('kbManager.columns.name', { defaultValue: 'Name' })}</TableHead>
                <TableHead className="text-center">{t('kbManager.columns.documents', { defaultValue: 'Documents' })}</TableHead>
                <TableHead className="text-center">{t('kbManager.columns.entities', { defaultValue: 'Entities' })}</TableHead>
                <TableHead>{t('kbManager.columns.status', { defaultValue: 'Status' })}</TableHead>
                <TableHead className="w-28 text-center">{t('kbManager.columns.actions', { defaultValue: 'Actions' })}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody className="text-sm">
              {rows.length === 0 && !isLoading && (
                <TableRow>
                  <TableCell colSpan={5} className="text-muted-foreground text-center py-8">
                    {t('kbManager.empty', { defaultValue: 'No knowledge bases found.' })}
                  </TableCell>
                </TableRow>
              )}
              {rows.map((row) => {
                const isCurrent = row.id === selectedKbId
                const isDefault = !!defaultKbId && row.id === defaultKbId
                return (
                  <TableRow key={row.id}>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{row.id}</span>
                        {isDefault && (
                          <Badge variant="outline">{t('kbManager.default', { defaultValue: 'default' })}</Badge>
                        )}
                        {isCurrent && (
                          <Badge variant="secondary" className="gap-1">
                            <CheckIcon className="size-3" />
                            {t('kbManager.inUse', { defaultValue: 'in use' })}
                          </Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-center">
                      {isLoading ? '-' : row.total ?? '-'}
                    </TableCell>
                    <TableCell className="text-center">
                      {isLoading ? '-' : row.entities ?? '-'}
                    </TableCell>
                    <TableCell>
                      {row.total === null ? (
                        <span className="text-muted-foreground text-xs">-</span>
                      ) : (
                        <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs">
                          <span className="text-green-600">
                            {t('documentPanel.documentManager.status.completed')} {row.processed}
                          </span>
                          <span className="text-blue-600">
                            {t('documentPanel.documentManager.status.processing')} {row.processing}
                          </span>
                          <span className="text-yellow-600">
                            {t('documentPanel.documentManager.status.pending')} {row.pending}
                          </span>
                          <span className="text-red-600">
                            {t('documentPanel.documentManager.status.failed')} {row.failed}
                          </span>
                        </div>
                      )}
                    </TableCell>
                    <TableCell className="text-center">
                      <div className="flex items-center justify-center gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 px-2"
                          disabled={isCurrent}
                          onClick={() => selectKnowledgeBase(row.id)}
                        >
                          {isCurrent
                            ? t('kbManager.switched', { defaultValue: 'Current' })
                            : t('kbManager.switch', { defaultValue: 'Use' })}
                        </Button>
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="size-7"
                              disabled={isDefault}
                              tooltip={isDefault
                                ? t('header.kbDeleteDefaultDisabled', { defaultValue: 'The default knowledge base cannot be deleted' })
                                : t('common.actions', { defaultValue: 'Actions' })}
                            >
                              <EllipsisIcon className="size-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuItem
                              destructive
                              onClick={() => {
                                setPendingDelete(row)
                                setDeleteConfirm('')
                              }}
                            >
                              <Trash2Icon />
                              {t('kbManager.delete', { defaultValue: 'Delete knowledge base' })}
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog open={isCreateOpen} onOpenChange={(open) => !isCreating && setIsCreateOpen(open)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{t('header.kbCreateTitle', { defaultValue: 'Create Knowledge Base' })}</DialogTitle>
            <DialogDescription>
              {t('kbManager.createDescription', {
                defaultValue: 'The identifier is used as the storage workspace, so keep it filesystem safe.'
              })}
            </DialogDescription>
          </DialogHeader>
          <Input
            value={newKbId}
            onChange={(e) => setNewKbId(e.target.value)}
            placeholder={t('header.kbCreatePlaceholder', { defaultValue: 'Enter knowledge base ID' })}
            disabled={isCreating}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsCreateOpen(false)} disabled={isCreating}>
              {t('common.cancel')}
            </Button>
            <Button onClick={handleCreate} disabled={isCreating}>
              {t('common.create', { defaultValue: 'Create' })}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <AlertDialog open={!!pendingDelete} onOpenChange={(open) => !isDeleting && !open && setPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2 text-red-500 dark:text-red-400">
              <TriangleAlertIcon className="size-5" aria-hidden="true" />
              {t('header.kbDeleteTitle', { defaultValue: 'Delete Knowledge Base' })}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {t('header.kbDeleteDescription', {
                defaultValue: 'Delete "{{kbId}}" and drop all of its storages. This cannot be undone.',
                kbId: pendingDelete?.id ?? ''
              })}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <div className="flex flex-col gap-2">
            {pendingDelete && (pendingDelete.total ?? 0) > 0 && (
              <p className="text-red-500 text-sm">
                {t('kbManager.deleteHasDocuments', {
                  defaultValue: 'This base still holds {{count}} document(s).',
                  count: pendingDelete.total ?? 0
                })}
              </p>
            )}
            <Input
              value={deleteConfirm}
              onChange={(e) => setDeleteConfirm(e.target.value)}
              placeholder={t('header.kbDeletePlaceholder', { defaultValue: 'Type "yes" to confirm' })}
              disabled={isDeleting}
            />
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('common.cancel')}</AlertDialogCancel>
            <AlertDialogAction onClick={handleDelete} disabled={deleteConfirm.toLowerCase() !== 'yes'}>
              {isDeleting
                ? t('header.kbDeleting', { defaultValue: 'Deleting' })
                : t('common.delete')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}
