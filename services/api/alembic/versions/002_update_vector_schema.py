"""update vector schema

Revision ID: 002
Revises: 001
Create Date: 2024-03-20 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Update documents table
    op.add_column('documents', sa.Column('doc_id', sa.String(255), nullable=False))
    op.add_column('documents', sa.Column('file_type', sa.String(50), nullable=False))
    op.add_column('documents', sa.Column('total_chunks', sa.Integer(), server_default='0'))
    op.create_unique_constraint('uq_documents_doc_id', 'documents', ['doc_id'])
    
    # Update crawl_jobs table
    op.add_column('crawl_jobs', sa.Column('job_id', sa.String(255), nullable=False))
    op.add_column('crawl_jobs', sa.Column('start_urls', sa.ARRAY(sa.Text()), nullable=False))
    op.add_column('crawl_jobs', sa.Column('pages_crawled', sa.Integer(), server_default='0'))
    op.add_column('crawl_jobs', sa.Column('pages_failed', sa.Integer(), server_default='0'))
    op.add_column('crawl_jobs', sa.Column('chunks_created', sa.Integer(), server_default='0'))
    op.add_column('crawl_jobs', sa.Column('filters', JSONB, server_default='{}'))
    op.add_column('crawl_jobs', sa.Column('completed_at', sa.DateTime(timezone=True)))
    op.add_column('crawl_jobs', sa.Column('next_scheduled', sa.DateTime(timezone=True)))
    op.add_column('crawl_jobs', sa.Column('error_details', sa.Text()))
    op.create_unique_constraint('uq_crawl_jobs_job_id', 'crawl_jobs', ['job_id'])
    
    # Create web_pages table
    op.create_table(
        'web_pages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('page_id', sa.String(255), nullable=False),
        sa.Column('crawl_job_id', sa.String(255), nullable=False),
        sa.Column('url', sa.Text(), nullable=False),
        sa.Column('title', sa.String(500)),
        sa.Column('domain', sa.String(255)),
        sa.Column('crawled_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('status', sa.String(50), server_default='success'),
        sa.Column('content_length', sa.Integer()),
        sa.Column('chunk_count', sa.Integer(), server_default='0'),
        sa.ForeignKeyConstraint(['crawl_job_id'], ['crawl_jobs.job_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('page_id')
    )
    
    # Create document_chunks table
    op.create_table(
        'document_chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('chunk_id', sa.String(255), nullable=False),
        sa.Column('doc_id', sa.String(255), nullable=False),
        sa.Column('page_id', sa.String(255)),
        sa.Column('chunk_text', sa.Text(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('page_number', sa.Integer()),
        sa.Column('section_title', sa.String(255)),
        sa.Column('content_type', sa.String(50), server_default='document'),
        sa.Column('source_url', sa.Text()),
        sa.Column('domain', sa.String(255)),
        sa.Column('faiss_index', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['doc_id'], ['documents.doc_id'], ),
        sa.ForeignKeyConstraint(['page_id'], ['web_pages.page_id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('chunk_id')
    )
    
    # Create chat_sessions table
    op.create_table(
        'chat_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(255), nullable=False),
        sa.Column('user_id', sa.String(255)),
        sa.Column('session_type', sa.String(50), server_default='anonymous'),
        sa.Column('ip_address', sa.String(50)),
        sa.Column('user_agent', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('last_activity', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('expires_at', sa.DateTime(timezone=True), server_default=sa.text("now() + interval '24 hours'")),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id')
    )
    
    # Create chat_messages table
    op.create_table(
        'chat_messages',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.String(255), nullable=False),
        sa.Column('message_text', sa.Text(), nullable=False),
        sa.Column('message_type', sa.String(50), nullable=False),
        sa.Column('faiss_index', sa.Integer()),
        sa.Column('response_time_ms', sa.Integer()),
        sa.Column('sources_used', JSONB),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.session_id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create processing_jobs table
    op.create_table(
        'processing_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.String(255), nullable=False),
        sa.Column('job_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), server_default='pending'),
        sa.Column('input_data', JSONB, nullable=False),
        sa.Column('result_data', JSONB),
        sa.Column('error_message', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('started_at', sa.DateTime(timezone=True)),
        sa.Column('completed_at', sa.DateTime(timezone=True)),
        sa.Column('retry_count', sa.Integer(), server_default='0'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id')
    )

def downgrade() -> None:
    # Drop processing_jobs table
    op.drop_table('processing_jobs')
    
    # Drop chat_messages table
    op.drop_table('chat_messages')
    
    # Drop chat_sessions table
    op.drop_table('chat_sessions')
    
    # Drop document_chunks table
    op.drop_table('document_chunks')
    
    # Drop web_pages table
    op.drop_table('web_pages')
    
    # Update crawl_jobs table
    op.drop_constraint('uq_crawl_jobs_job_id', 'crawl_jobs', type_='unique')
    op.drop_column('crawl_jobs', 'error_details')
    op.drop_column('crawl_jobs', 'next_scheduled')
    op.drop_column('crawl_jobs', 'completed_at')
    op.drop_column('crawl_jobs', 'filters')
    op.drop_column('crawl_jobs', 'chunks_created')
    op.drop_column('crawl_jobs', 'pages_failed')
    op.drop_column('crawl_jobs', 'pages_crawled')
    op.drop_column('crawl_jobs', 'start_urls')
    op.drop_column('crawl_jobs', 'job_id')
    
    # Update documents table
    op.drop_constraint('uq_documents_doc_id', 'documents', type_='unique')
    op.drop_column('documents', 'total_chunks')
    op.drop_column('documents', 'file_type')
    op.drop_column('documents', 'doc_id') 