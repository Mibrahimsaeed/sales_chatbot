# """add chat_log table

# Revision ID: 0003
# Revises: 0002
# """

# import sqlalchemy as sa
# from alembic import op

# revision = "0003"
# down_revision = "0002"
# branch_labels = None
# depends_on = None


# def upgrade():
#     op.create_table(
#         "chat_log",
#         sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
#         sa.Column("session_id", sa.String, index=True),
#         sa.Column("user_message", sa.String, nullable=False),
#         sa.Column("detected_intent", sa.String),
#         sa.Column("confidence", sa.Float),
#         sa.Column("used_llm_fallback", sa.Boolean, server_default=sa.false()),
#         sa.Column("response_type", sa.String),
#         sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
#     )


# def downgrade():
#     op.drop_table("chat_log")

"""add chat_log table

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op


revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "chat_log",
        sa.Column(
            "id",
            sa.Integer,
            primary_key=True,
            autoincrement=True
        ),
        sa.Column(
            "session_id",
            sa.String,
            index=True
        ),
        sa.Column(
            "user_message",
            sa.String,
            nullable=False
        ),
        sa.Column(
            "detected_intent",
            sa.String
        ),
        sa.Column(
            "confidence",
            sa.Float
        ),
        sa.Column(
            "used_llm_fallback",
            sa.Boolean,
            server_default=sa.false()
        ),
        sa.Column(
            "response_type",
            sa.String
        ),
        sa.Column(
            "created_at",
            sa.DateTime,
            server_default=sa.func.now()
        ),
    )


def downgrade():
    op.drop_table("chat_log")