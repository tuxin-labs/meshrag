import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { FilePlus2Icon } from 'lucide-react'
import Button from '@/components/ui/Button'
import Checkbox from '@/components/ui/Checkbox'
import Input from '@/components/ui/Input'
import Textarea from '@/components/ui/Textarea'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog'
import { insertText } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'

interface InsertTextDialogProps {
  onDocumentsInserted?: () => Promise<void> | void
  /** Semi-controlled open state; omit to let the dialog manage its own trigger. */
  open?: boolean
  onOpenChange?: (open: boolean) => void
}

/** Direct text ingestion, the counterpart to uploading a file. */
export default function InsertTextDialog({
  onDocumentsInserted,
  open: controlledOpen,
  onOpenChange
}: InsertTextDialogProps) {
  const { t } = useTranslation()
  const [internalOpen, setInternalOpen] = useState(false)
  const isControlled = controlledOpen !== undefined
  const open = isControlled ? controlledOpen : internalOpen
  const setOpen = (next: boolean) => (isControlled ? onOpenChange?.(next) : setInternalOpen(next))
  const [text, setText] = useState('')
  const [fileSource, setFileSource] = useState('')
  const [overwrite, setOverwrite] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async () => {
    if (text.trim().length < 1) {
      toast.error(t('documentPanel.insertText.empty', { defaultValue: 'Text is required' }))
      return
    }

    setIsSubmitting(true)
    try {
      const result = await insertText(text, fileSource.trim() || undefined, overwrite)
      if (result.status === 'duplicated') {
        toast.warning(result.original_file_path
          ? t('documentPanel.uploadDocuments.fileUploader.duplicateFileOf', { path: result.original_file_path })
          : t('documentPanel.uploadDocuments.fileUploader.duplicateFile'))
      } else if (result.status !== 'success') {
        toast.error(result.message)
      } else {
        toast.success(t('documentPanel.insertText.success', { defaultValue: 'Text submitted for indexing' }))
        setText('')
        setFileSource('')
        setOpen(false)
        await onDocumentsInserted?.()
      }
    } catch (error) {
      toast.error(errorMessage(error))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={(next) => !isSubmitting && setOpen(next)}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" side="bottom" tooltip={t('documentPanel.insertText.tooltip')}>
          <FilePlus2Icon /> {t('documentPanel.insertText.button')}
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('documentPanel.insertText.title')}</DialogTitle>
          <DialogDescription>{t('documentPanel.insertText.description')}</DialogDescription>
        </DialogHeader>

        <Input
          value={fileSource}
          onChange={(e) => setFileSource(e.target.value)}
          placeholder={t('documentPanel.insertText.sourcePlaceholder')}
          disabled={isSubmitting}
        />
        <Textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={t('documentPanel.insertText.textPlaceholder')}
          className="min-h-48"
          disabled={isSubmitting}
        />
        <label className="flex cursor-pointer items-center gap-2 text-sm">
          <Checkbox
            checked={overwrite}
            onCheckedChange={(checked) => setOverwrite(checked === true)}
            disabled={isSubmitting}
          />
          <span className="text-muted-foreground">{t('documentPanel.uploadDocuments.overwrite')}</span>
        </label>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} disabled={isSubmitting}>
            {t('common.cancel')}
          </Button>
          <Button onClick={handleSubmit} disabled={isSubmitting}>
            {isSubmitting
              ? t('documentPanel.insertText.submitting')
              : t('documentPanel.insertText.confirm')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
