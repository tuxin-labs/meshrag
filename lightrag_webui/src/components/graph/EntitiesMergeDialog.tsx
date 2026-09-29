import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
import Input from '@/components/ui/Input'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from '@/components/ui/Dialog'
import { mergeEntities } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { useGraphStore } from '@/stores/graph'
import { GitMergeIcon } from 'lucide-react'

const splitEntities = (raw: string): string[] =>
  Array.from(
    new Set(
      raw
        .split(/[,\n]/)
        .map((name) => name.trim())
        .filter(Boolean)
    )
  )

export default function EntitiesMergeDialog() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [sources, setSources] = useState('')
  const [target, setTarget] = useState('')

  const resetForm = () => {
    setSources('')
    setTarget('')
  }

  const handleMerge = async () => {
    const entitiesToChange = splitEntities(sources)
    const entityToChangeInto = target.trim()

    if (entitiesToChange.length === 0 || !entityToChangeInto) {
      toast.error(t('graphPanel.mergeEntities.required', {
        defaultValue: 'Select at least one entity to merge and a target entity'
      }))
      return
    }
    if (entitiesToChange.includes(entityToChangeInto)) {
      toast.error(t('graphPanel.mergeEntities.targetIsSource', {
        defaultValue: 'The target entity cannot also be merged'
      }))
      return
    }

    setIsSubmitting(true)
    try {
      await mergeEntities({
        entities_to_change: entitiesToChange,
        entity_to_change_into: entityToChangeInto
      })
      toast.success(t('graphPanel.mergeEntities.success', {
        defaultValue: '{{count}} entities merged',
        count: entitiesToChange.length
      }))
      useGraphStore.getState().incrementGraphDataVersion()
      resetForm()
      setOpen(false)
    } catch (err) {
      toast.error(errorMessage(err))
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (isSubmitting) return
        setOpen(next)
        if (!next) resetForm()
      }}
    >
      <DialogTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          side="right"
          tooltip={t('graphPanel.mergeEntities.tooltip', { defaultValue: 'Merge entities' })}
        >
          <GitMergeIcon className="size-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('graphPanel.mergeEntities.title', { defaultValue: 'Merge Entities' })}</DialogTitle>
          <DialogDescription>
            {t('graphPanel.mergeEntities.description', {
              defaultValue: 'Relationships of the source entities move into the target entity, which is kept.'
            })}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-2">
          <Input
            value={sources}
            onChange={(e) => setSources(e.target.value)}
            placeholder={t('graphPanel.mergeEntities.sourcesPlaceholder', {
              defaultValue: 'Entities to merge, separated by commas'
            })}
            disabled={isSubmitting}
          />
          <Input
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            placeholder={t('graphPanel.mergeEntities.targetPlaceholder', { defaultValue: 'Target entity' })}
            disabled={isSubmitting}
          />
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} disabled={isSubmitting}>
            {t('common.cancel')}
          </Button>
          <Button onClick={handleMerge} disabled={isSubmitting}>
            {isSubmitting
              ? t('graphPanel.mergeEntities.merging', { defaultValue: 'Merging' })
              : t('graphPanel.mergeEntities.confirm', { defaultValue: 'Merge' })}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
