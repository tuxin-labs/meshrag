import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import Checkbox from '@/components/ui/Checkbox'
import Textarea from '@/components/ui/Textarea'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/Dialog'
import { errorMessage } from '@/lib/utils'
import { updateDocumentText } from '@/api/lightrag'
import type { DocStatusResponse } from '@/api/lightrag'

interface EditDocumentTextDialogProps {
  document: DocStatusResponse | null
  onOpenChange: (open: boolean) => void
  onUpdated?: () => Promise<void> | void
}

/**
 * Replaces a document's content. The backend locates the document by file_source, so the
 * full text has to be supplied again — DocStatusResponse only carries a summary.
 */
export default function EditDocumentTextDialog({
  document,
  onOpenChange,
  onUpdated
}: EditDocumentTextDialogProps) {
  const { t } = useTranslation()
  const [text, setText] = useState('')
  const [deleteLlmCache, setDeleteLlmCache] = useState(true)
  const [isSaving, setIsSaving] = useState(false)

  useEffect(() => {
    setText('')
    setDeleteLlmCache(true)
  }, [document?.id])

  const handleSave = async () => {
    if (!document) return
    if (!text.trim()) {
      toast.error(t('documentPanel.editDocument.emptyText', { defaultValue: 'Text is required' }))
      return
    }
    if (!document.file_path) {
      toast.error(t('documentPanel.editDocument.noFileSource', {
        defaultValue: 'This document has no file source, so it cannot be updated'
      }))
      return
    }

    setIsSaving(true)
    try {
      const result = await updateDocumentText({
        text,
        file_source: document.file_path,
        delete_llm_cache: deleteLlmCache
      })

      if (result.status === 'success') {
        toast.success(t('documentPanel.editDocument.success', { defaultValue: result.message }))
        onOpenChange(false)
        await onUpdated?.()
      } else if (result.status === 'unchanged') {
        toast.info(t('documentPanel.editDocument.unchanged', { defaultValue: result.message }))
      } else if (result.status === 'not_found') {
        toast.error(t('documentPanel.editDocument.notFound', { defaultValue: result.message }))
      } else {
        toast.error(t('documentPanel.editDocument.failed', { defaultValue: result.message }))
      }
    } catch (err) {
      toast.error(t('documentPanel.editDocument.error', { error: errorMessage(err) }))
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <Dialog open={!!document} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('documentPanel.editDocument.title', { defaultValue: 'Update document' })}</DialogTitle>
          <DialogDescription className="break-all">
            {t('documentPanel.editDocument.description', {
              defaultValue: 'Replace the content indexed for this file source.'
            })}
            {document?.file_path && <span className="mt-1 block font-mono text-xs">{document.file_path}</span>}
          </DialogDescription>
        </DialogHeader>

        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={t('documentPanel.editDocument.placeholder', {
            defaultValue: 'Paste the new full text of the document'
          })}
          className="min-h-48"
          disabled={isSaving}
        />

        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <Checkbox
            checked={deleteLlmCache}
            onCheckedChange={(checked) => setDeleteLlmCache(checked === true)}
            disabled={isSaving}
          />
          <span className="text-muted-foreground">
            {t('documentPanel.editDocument.deleteCache', { defaultValue: 'Drop the LLM cache of this document' })}
          </span>
        </label>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isSaving}>
            {t('common.cancel')}
          </Button>
          <Button onClick={handleSave} disabled={isSaving}>
            {isSaving
              ? t('documentPanel.editDocument.saving', { defaultValue: 'Saving' })
              : t('common.save', { defaultValue: 'Save' })}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
