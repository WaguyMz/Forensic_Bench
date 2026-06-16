DB_SCHEMA = """
## Database Schema

### Postgres schema introspection (use these; do NOT invent columns)
- To list columns/types (portable): `SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_schema='public' AND table_name = '<table>' ORDER BY ordinal_position;`
- To list tables: `SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name;`
- Notes:
  - `information_schema.columns` in Postgres does **not** have fields like `is_key` or `column_comment`.
  - If you need key/constraint info, use `pg_catalog` (e.g. `pg_constraint`) rather than querying non-existent `information_schema` fields.

### `je_header`
| Column           | Type    | Notes                                           |
|------------------|---------|-------------------------------------------------|
| document_id      | UUID PK | Unique journal entry identifier                 |
| company_code     | text    | Legal entity code                               |
| fiscal_year      | int     | Fiscal year                                     |
| fiscal_period    | int     | Period within the year (1-12)                   |
| posting_date     | date    | Date posted to the ledger                       |
| document_date    | date    | Original document / invoice date                |
| document_type    | text    | JE type (e.g. SA, KR, KZ, RV, AB …)            |
| currency         | text    | Transaction currency (ISO 4217)                 |
| exchange_rate    | numeric | FX rate to local currency                       |
| reference        | text    | External reference / invoice number             |
| created_by       | text    | User who posted the entry                       |
| source           | text    | Source system / module                          |
| business_process | text    | Business process tag (P2P, O2C, HR, …)         |
| ledger           | text    | General ledger / sub-ledger ID                  |

### `je_line`
| Column                   | Type    | Notes                                       |
|--------------------------|---------|---------------------------------------------|
| document_id              | UUID FK | Links to je_header                          |
| line_number              | int     | Line sequence within the document           |
| company_code             | text    |                                             |
| gl_account               | text    | GL account number                           |
| debit_amount             | numeric |                                             |
| credit_amount            | numeric |                                             |
| local_amount             | numeric | Signed net amount in local currency         |
| cost_center              | text    |                                             |
| profit_center            | text    |                                             |
| auxiliary_account_number | text    | Sub-ledger counterparty account reference   |
| auxiliary_account_label  | text    | Display name for the sub-ledger account     |
| lettrage                 | text    | Matching / lettering code (FEC convention)  |
| lettrage_date            | date    | Date the line was matched                   |

### `employees`
| Column                 | Type    | Notes                              |
|------------------------|---------|------------------------------------|
| employee_id            | text PK |                                    |
| user_id                | text    | Login / system user ID             |
| display_name           | text    |                                    |
| first_name / last_name | text    |                                    |
| email                  | text    |                                    |
| company_code           | text    |                                    |
| department_id          | text    |                                    |
| cost_center            | text    |                                    |
| manager_id             | text    | FK to another employee_id          |
| status                 | text    | ACTIVE, TERMINATED                 |
| hire_date              | date    |                                    |
| termination_date       | date    | Null if still active               |
| creation_date          | date    |                                    |
| location               | text    |                                    |
| payroll_bank_name      | text    |                                    |
| payroll_bank_country   | text    |                                    |
| payroll_account_number | text    | Bank account used for salary       |
| payroll_routing_code   | text    |                                    |

### `hr_employees`
**HR department's official employee registry** — excludes ghost employees and fraud actors.
Use this table to detect shadow payroll schemes: employees present in `employees` but
missing from `hr_employees` are potential ghost employees.

| Column                 | Type    | Notes                              |
|------------------------|---------|------------------------------------|
| employee_id            | text PK |                                    |
| user_id                | text    | Login / system user ID             |
| display_name           | text    |                                    |
| first_name / last_name | text    |                                    |
| email                  | text    |                                    |
| company_code           | text    |                                    |
| department_id          | text    |                                    |
| cost_center            | text    |                                    |
| manager_id             | text    | FK to another employee_id          |
| status                 | text    | ACTIVE, TERMINATED                 |
| hire_date              | date    |                                    |
| termination_date       | date    | Null if still active               |
| creation_date          | date    |                                    |
| location               | text    |                                    |


### `vendors`
| Column                 | Type    | Notes                              |
|------------------------|---------|------------------------------------|
| vendor_id              | text PK |                                    |
| name                   | text    |                                     |
| country                | text    |                                     |
| account_number         | text    |                                     |
| tax_id                 | text    |                                     |
| currency               | text    |                                     |
| reconciliation_account | text    | Accounts-payable control account   |
| auxiliary_gl_account   | text    | Vendor sub-ledger account          |
| is_intercompany        | bool    |                                     |
| behavior               | text    | regular, sparse, high_value, …     |
| payment_terms          | text    |                                     |
| bank_account_count     | int     |                                     |
| primary_bank_name      | text    |                                     |
| primary_bank_country   | text    |                                     |
| primary_account_number | text    | Primary bank account               |
| primary_routing_code   | text    |                                     |

### `customers`
| Column                 | Type    | Notes                              |
|------------------------|---------|------------------------------------|
| customer_id            | text PK |                                    |
| name                   | text    |                                     |
| country                | text    |                                     |
| account_number         | text    |                                     |
| tax_id                 | text    |                                     |
| currency               | text    |                                     |
| reconciliation_account | text    | Accounts-receivable control account|
| auxiliary_gl_account   | text    | Customer sub-ledger account        |
| is_intercompany        | bool    |                                     |
| credit_rating          | text    |                                     |
| bank_account_count     | int     |                                     |
| primary_bank_name      | text    |                                     |
| primary_bank_country   | text    |                                     |
| primary_account_number | text    | Primary bank account               |
| primary_routing_code   | text    |                                     |

### `chart_of_accounts`
Full Plan Comptable Général (PCG 2024) hierarchy — every account and intermediate
group in the French chart of accounts.  Join to `je_line.gl_account` to resolve
account labels and navigate the class/subclass structure.

| Column                | Type    | Notes                                                        |
|-----------------------|---------|--------------------------------------------------------------|
| account_number        | text PK | 6-digit normalised GL code (e.g. "603000", "411000")         |
| label                 | text    | Official French PCG label (e.g. "Achats stockés — mat. 1ères") |
| parent_account_number | text    | Parent node's 6-digit code; NULL for the 8 class-level nodes |
| account_type          | text    | Simplified class type: Asset · Liability · Equity · Tiers · Expense · Revenue · Special |
| level                 | int     | Hierarchy depth — 1 = class (1-digit), 2 = subclass, 3 = group, 4+ = account |

"""
