//! DDL statement AST: CREATE TABLE / INDEX / VIEW / TRIGGER, DROP, ALTER TABLE.
//! Specs: `lang_createtable.html`, `lang_createindex.html`, `lang_createview.html`,
//! `lang_createtrigger.html`, `lang_altertable.html`, `lang_droptable.html`.

use crate::ast_expr::{Expr, Literal};
use crate::ast_select::{Select, SortOrder};
use crate::ast_stmt::{ConflictClause, QualifiedName, Statement};

// ---------------------------------------------------------------------------
// CREATE TABLE
// ---------------------------------------------------------------------------

/// `CREATE [TEMP] TABLE [IF NOT EXISTS] name <body>`.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTable {
    pub temp: bool,
    pub if_not_exists: bool,
    pub name: QualifiedName,
    pub body: CreateTableBody,
}

/// The two shapes of a table definition.
#[derive(Debug, Clone, PartialEq)]
pub enum CreateTableBody {
    /// `(column-def, ..., table-constraint, ...) [WITHOUT ROWID] [, STRICT]`.
    Columns { columns: Vec<ColumnDef>, constraints: Vec<TableConstraint>, options: TableOptions },
    /// `AS select-stmt`.
    AsSelect(Box<Select>),
}

/// Trailing table options after the column list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TableOptions {
    pub without_rowid: bool,
    pub strict: bool,
}

/// One column definition: `name [type] [constraints...]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnDef {
    pub name: String,
    /// The declared type name, verbatim (affinity is derived later). `None` when
    /// the column has no type (SQLite allows that).
    pub type_name: Option<String>,
    pub constraints: Vec<ColumnConstraint>,
}

/// A column constraint, with its optional `CONSTRAINT name` label.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnConstraint {
    pub name: Option<String>,
    pub kind: ColumnConstraintKind,
}

/// The kinds of column-level constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnConstraintKind {
    PrimaryKey { order: Option<SortOrder>, conflict: Option<ConflictClause>, autoincrement: bool },
    NotNull { conflict: Option<ConflictClause> },
    /// A bare `NULL` constraint (accepted by SQLite; imposes nothing).
    Null { conflict: Option<ConflictClause> },
    Unique { conflict: Option<ConflictClause> },
    Check(Expr),
    Default(DefaultValue),
    Collate(String),
    ForeignKey(ForeignKeyClause),
    /// `[GENERATED ALWAYS] AS (expr) [STORED|VIRTUAL]`. `stored` is false for VIRTUAL.
    Generated { expr: Expr, stored: bool },
}

/// A column `DEFAULT` value.
#[derive(Debug, Clone, PartialEq)]
pub enum DefaultValue {
    /// A literal or signed number (sign already folded into the literal).
    Literal(Literal),
    /// A parenthesized expression: `DEFAULT (expr)`, with the source text between the
    /// parentheses (outer whitespace trimmed). That text is what sqlite3 reports as
    /// `dflt_value` and what the INSERT planner re-parses to apply the default.
    Expr(Box<Expr>, String),
}

/// A table-level constraint, with its optional `CONSTRAINT name` label.
#[derive(Debug, Clone, PartialEq)]
pub struct TableConstraint {
    pub name: Option<String>,
    pub kind: TableConstraintKind,
}

/// The kinds of table-level constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum TableConstraintKind {
    PrimaryKey { columns: Vec<IndexedColumn>, conflict: Option<ConflictClause> },
    Unique { columns: Vec<IndexedColumn>, conflict: Option<ConflictClause> },
    Check(Expr),
    /// `FOREIGN KEY (cols) REFERENCES ...`.
    ForeignKey { columns: Vec<String>, clause: ForeignKeyClause },
}

/// A `REFERENCES` foreign-key clause (used by both a column FK and a table FK).
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignKeyClause {
    pub table: String,
    /// The referenced columns (empty = the referenced table's primary key).
    pub columns: Vec<String>,
    pub actions: Vec<ForeignKeyAction>,
    /// `MATCH name` (SQLite parses but ignores it).
    pub match_name: Option<String>,
    pub deferrable: Option<Deferrable>,
}

/// An `ON DELETE` / `ON UPDATE` foreign-key action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForeignKeyAction {
    OnDelete(ReferentialAction),
    OnUpdate(ReferentialAction),
}

/// What a foreign-key action does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferentialAction {
    SetNull,
    SetDefault,
    Cascade,
    Restrict,
    NoAction,
}

/// `[NOT] DEFERRABLE [INITIALLY DEFERRED|IMMEDIATE]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Deferrable {
    pub not: bool,
    pub initially: Option<InitiallyTiming>,
}

/// `INITIALLY DEFERRED` / `INITIALLY IMMEDIATE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitiallyTiming {
    Deferred,
    Immediate,
}

/// One column in an index / PK / UNIQUE column list: a name or an expression, with
/// optional collation and sort order.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexedColumn {
    pub target: IndexedColumnTarget,
    pub collation: Option<String>,
    pub order: Option<SortOrder>,
}

/// An indexed column is either a plain column name or an expression.
#[derive(Debug, Clone, PartialEq)]
pub enum IndexedColumnTarget {
    Name(String),
    Expr(Expr),
}

// ---------------------------------------------------------------------------
// CREATE INDEX
// ---------------------------------------------------------------------------

/// `CREATE [UNIQUE] INDEX [IF NOT EXISTS] name ON table (columns) [WHERE expr]`.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateIndex {
    pub unique: bool,
    pub if_not_exists: bool,
    pub name: QualifiedName,
    pub table: String,
    pub columns: Vec<IndexedColumn>,
    pub where_clause: Option<Expr>,
}

// ---------------------------------------------------------------------------
// CREATE VIEW
// ---------------------------------------------------------------------------

/// `CREATE [TEMP] VIEW [IF NOT EXISTS] name [(columns)] AS select`.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateView {
    pub temp: bool,
    pub if_not_exists: bool,
    pub name: QualifiedName,
    pub columns: Option<Vec<String>>,
    pub select: Box<Select>,
}

// ---------------------------------------------------------------------------
// CREATE TRIGGER
// ---------------------------------------------------------------------------

/// A `CREATE TRIGGER` statement. The body is a list of statements (INSERT / UPDATE
/// / DELETE / SELECT) parsed elsewhere; the struct is defined now.
///
/// `table` is the `ON`-clause TARGET, a possibly schema-qualified name. The spec's
/// "TEMP Triggers on Non-TEMP Tables" (`lang_createtrigger.html` §7) recommends the
/// qualified form (`CREATE TEMP TRIGGER … ON main.tab …`), so the target carries an
/// optional schema qualifier just like the trigger's own `name`; an unqualified target
/// resolves by search order at create time.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateTrigger {
    pub temp: bool,
    pub if_not_exists: bool,
    pub name: QualifiedName,
    pub timing: Option<TriggerTiming>,
    pub event: TriggerEvent,
    pub table: QualifiedName,
    pub for_each_row: bool,
    pub when: Option<Expr>,
    pub body: Vec<Statement>,
}

/// When a trigger fires relative to its event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerTiming {
    Before,
    After,
    InsteadOf,
}

/// The event a trigger fires on.
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerEvent {
    Delete,
    Insert,
    /// `UPDATE [OF col, col, ...]`.
    Update { columns: Vec<String> },
}

// ---------------------------------------------------------------------------
// DROP
// ---------------------------------------------------------------------------

/// `DROP {TABLE|INDEX|VIEW|TRIGGER} [IF EXISTS] name`.
#[derive(Debug, Clone, PartialEq)]
pub struct Drop {
    pub kind: DropKind,
    pub if_exists: bool,
    pub name: QualifiedName,
}

/// The object kind a `DROP` targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropKind {
    Table,
    Index,
    View,
    Trigger,
}

// ---------------------------------------------------------------------------
// ALTER TABLE
// ---------------------------------------------------------------------------

/// `ALTER TABLE name <action>`.
#[derive(Debug, Clone, PartialEq)]
pub struct AlterTable {
    pub name: QualifiedName,
    pub action: AlterAction,
}

/// The alteration performed by `ALTER TABLE`.
#[derive(Debug, Clone, PartialEq)]
pub enum AlterAction {
    RenameTo(String),
    RenameColumn { from: String, to: String },
    AddColumn(ColumnDef),
    DropColumn(String),
}
