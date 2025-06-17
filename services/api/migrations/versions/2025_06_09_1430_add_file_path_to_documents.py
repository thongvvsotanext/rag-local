"""Add file_path to documents

Revision ID: 2025_06_09_1430
Revises: 
Create Date: 2025-06-09 14:30:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '2025_06_09_1430'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Add file_path column to documents table
    op.add_column('documents', sa.Column('file_path', sa.String(), nullable=True))
    # Create index on file_path for faster lookups
    op.create_index(op.f('ix_documents_file_path'), 'documents', ['file_path'], unique=False)

def downgrade():
    # Drop the index first
    op.drop_index(op.f('ix_documents_file_path'), table_name='documents')
    # Then drop the column
    op.drop_column('documents', 'file_path')
