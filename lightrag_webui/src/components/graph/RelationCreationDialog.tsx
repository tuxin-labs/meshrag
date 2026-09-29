import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import Button from '@/components/ui/Button'
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
import { createRelation } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { useGraphStore } from '@/stores/graph'
import { GitBranchPlusIcon } from 'lucide-react'

export default function RelationCreationDialog() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [sourceEntity, setSourceEntity] = useState('')
  const [targetEntity, setTargetEntity] = useState('')
  const [keywords, setKeywords] = useState('')
  const [description, setDescription] = useState('')
  const [weight, setWeight] = useState('1.0')

  const resetForm = () => {
    setSourceEntity('')
    setTargetEntity('')
    setKeywords('')
    setDescription('')
    setWeight('1.0')
  }

  const handleCreate = async () => {
    if (!sourceEntity.trim() || !targetEntity.trim()) {
      toast.error(t('graphPanel.createRelation.entitiesRequired', {
        defaultValue: 'Both entities are required'
      }))
      return
    }

    setIsSubmitting(true)
    try {
      await createRelation({
        source_entity: sourceEntity.trim(),
        target_entity: targetEntity.trim(),
        relation_data: {
          description: description.trim(),
          keywords: keywords.trim(),
          weight: parseFloat(weight) || 1.0
        }
      })
      toast.success(t('graphPanel.createRelation.success', { defaultValue: 'Relation created' }))
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
          tooltip={t('graphPanel.createRelation.tooltip', { defaultValue: 'Create relation' })}
        >
          <GitBranchPlusIcon className="size-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('graphPanel.createRelation.title', { defaultValue: 'Create Relation' })}</DialogTitle>
          <DialogDescription>
            {t('graphPanel.createRelation.description', {
              defaultValue: 'Both entities must already exist in the knowledge graph.'
            })}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-2">
          <Input
            value={sourceEntity}
            onChange={(e) => setSourceEntity(e.target.value)}
            placeholder={t('graphPanel.propertiesView.edge.source')}
            disabled={isSubmitting}
          />
          <Input
            value={targetEntity}
            onChange={(e) => setTargetEntity(e.target.value)}
            placeholder={t('graphPanel.propertiesView.edge.target')}
            disabled={isSubmitting}
          />
          <Input
            value={keywords}
            onChange={(e) => setKeywords(e.target.value)}
            placeholder={t('graphPanel.createRelation.keywordsPlaceholder', { defaultValue: 'Keywords' })}
            disabled={isSubmitting}
          />
          <Input
            type="number"
            step="0.1"
            value={weight}
            onChange={(e) => setWeight(e.target.value)}
            placeholder={t('graphPanel.propertiesView.node.propertyNames.weight', { defaultValue: 'Weight' })}
            className="w-32"
            disabled={isSubmitting}
          />
          <Textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder={t('graphPanel.createRelation.descriptionPlaceholder', { defaultValue: 'Description' })}
            className="min-h-20"
            disabled={isSubmitting}
          />
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} disabled={isSubmitting}>
            {t('common.cancel')}
          </Button>
          <Button onClick={handleCreate} disabled={isSubmitting}>
            {isSubmitting ? t('common.saving') : t('common.save')}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
