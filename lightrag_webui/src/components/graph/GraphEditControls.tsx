import EntityCreationDialog from './EntityCreationDialog'
import RelationCreationDialog from './RelationCreationDialog'
import EntitiesMergeDialog from './EntitiesMergeDialog'

/** Graph mutation actions that operate on the knowledge base currently selected. */
export default function GraphEditControls() {
  return (
    <>
      <EntityCreationDialog />
      <RelationCreationDialog />
      <EntitiesMergeDialog />
    </>
  )
}
