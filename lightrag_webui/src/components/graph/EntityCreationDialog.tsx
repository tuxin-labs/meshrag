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
import { createEntity } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { useGraphStore } from '@/stores/graph'
import { UserPlusIcon } from 'lucide-react'

export default function EntityCreationDialog() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [entityName, setEntityName] = useState('')
  const [entityType, setEntityType] = useState('')
  const [description, setDescription] = useState('')

  const resetForm = () => {
    setEntityName('')
    setEntityType('')
    setDescription('')
  }

  const handleCreate = async () => {
    if (!entityName.trim()) {
      toast.error(t('graphPanel.createEntity.nameRequired', { defaultValue: 'Entity name is required' }))
      return
    }

    setIsSubmitting(true)
    try {
      await createEntity({
        entity_name: entityName.trim(),
        entity_data: {
          entity_type: entityType.trim() || 'UNKNOWN',
          description: description.trim()
        }
      })
      toast.success(t('graphPanel.createEntity.success', { defaultValue: 'Entity created' }))
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
          tooltip={t('graphPanel.createEntity.tooltip', { defaultValue: 'Create entity' })}
        >
          <UserPlusIcon className="size-4" />
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md" onCloseAutoFocus={(e) => e.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{t('graphPanel.createEntity.title', { defaultValue: 'Create Entity' })}</DialogTitle>
          <DialogDescription>
            {t('graphPanel.createEntity.description', {
              defaultValue: 'Add a node to the knowledge graph of the current knowledge base.'
            })}
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-2">
          <Input
            value={entityName}
            onChange={(e) => setEntityName(e.target.value)}
            placeholder={t('graphPanel.createEntity.namePlaceholder', { defaultValue: 'Entity name' })}
            disabled={isSubmitting}
          />
          <Input
            value={entityType}
            onChange={(e) => setEntityType(e.target.value)}
            placeholder={t('graphPanel.createEntity.typePlaceholder', { defaultValue: 'Entity type' })}
            disabled={isSubmitting}
          />
          <Textarea
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder={t('graphPanel.createEntity.descriptionPlaceholder', { defaultValue: 'Description' })}
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
