//! DDL parsing: CREATE {TABLE, INDEX, VIEW, TRIGGER}, DROP, ALTER TABLE.
//! Specs: `lang_createtable.html`, `lang_createindex.html`, `lang_createview.html`,
//! `lang_createtrigger.html`, `lang_altertable.html`, `lang_droptable.html`.

use crate::ast_ddl::{
    AlterAction, AlterTable, ColumnConstraint, ColumnConstraintKind, ColumnDef, CreateIndex,
    CreateTable, CreateTableBody, CreateTrigger, CreateView, DefaultValue, Deferrable, DropKind,
    ForeignKeyAction, ForeignKeyClause, IndexedColumn, IndexedColumnTarget, InitiallyTiming,
    ReferentialAction, TableConstraint, TableConstraintKind, TableOptions, TriggerEvent,
    TriggerTiming,
};
use crate::ast_expr::{Expr, Literal};
use crate::ast_select::SortOrder;
use crate::ast_stmt::{ConflictClause, Statement};
use crate::keyword::Keyword;
use crate::parser::Parser;
use crate::token::{QuoteKind, TokenKind};
use minisqlite_types::Result;

impl<'a> Parser<'a> {
    // --- CREATE dispatch ------------------------------------------------------

    pub(crate) fn parse_create(&mut self) -> Result<Statement> {
        self.expect_kw(Keyword::Create)?;
        let temp = self.eat_kw(Keyword::Temp) || self.eat_kw(Keyword::Temporary);

        if self.check_kw(Keyword::Table) {
            return self.parse_create_table(temp);
        }
        if self.check_kw(Keyword::View) {
            return self.parse_create_view(temp);
        }
        if self.check_kw(Keyword::Trigger) {
            return self.parse_create_trigger(temp);
        }
        if self.check_kw(Keyword::Unique) || self.check_kw(Keyword::Index) {
            if temp {
                return self.error("TEMP is not valid with CREATE INDEX");
            }
            return self.parse_create_index();
        }
        if self.check_kw(Keyword::Virtual) {
            return self.unsupported("CREATE VIRTUAL TABLE");
        }
        self.error("expected TABLE, INDEX, VIEW, TRIGGER or VIRTUAL after CREATE")
    }

    fn parse_if_not_exists(&mut self) -> Result<bool> {
        if self.eat_kw(Keyword::If) {
            self.expect_kw(Keyword::Not)?;
            self.expect_kw(Keyword::Exists)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // --- CREATE TABLE ---------------------------------------------------------

    fn parse_create_table(&mut self, temp: bool) -> Result<Statement> {
        self.expect_kw(Keyword::Table)?;
        let if_not_exists = self.parse_if_not_exists()?;
        let name = self.parse_qualified_name()?;
        let body = if self.eat_kw(Keyword::As) {
            CreateTableBody::AsSelect(Box::new(self.parse_select()?))
        } else {
            self.parse_table_columns_body()?
        };
        Ok(Statement::CreateTable(Box::new(CreateTable { temp, if_not_exists, name, body })))
    }

    fn parse_table_columns_body(&mut self) -> Result<CreateTableBody> {
        self.expect(&TokenKind::LParen, "'(' or AS in CREATE TABLE")?;
        let mut columns = Vec::new();
        let mut constraints = Vec::new();
        loop {
            if self.at_table_constraint_start() {
                constraints.push(self.parse_table_constraint()?);
            } else {
                columns.push(self.parse_column_def()?);
            }
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        self.expect(&TokenKind::RParen, "')' after CREATE TABLE column list")?;
        let options = self.parse_table_options()?;
        Ok(CreateTableBody::Columns { columns, constraints, options })
    }

    fn at_table_constraint_start(&self) -> bool {
        matches!(
            self.kw(),
            Some(
                Keyword::Constraint
                    | Keyword::Primary
                    | Keyword::Unique
                    | Keyword::Check
                    | Keyword::Foreign
            )
        )
    }

    fn parse_table_options(&mut self) -> Result<TableOptions> {
        // `ROWID` is not a keyword in SQLite — it is an ordinary identifier — so
        // `WITHOUT ROWID` is the `WITHOUT` keyword followed by the ident `rowid`.
        // `STRICT` likewise is an identifier, not a keyword.
        let mut opts = TableOptions::default();
        loop {
            if self.check_kw(Keyword::Without) {
                self.bump();
                if !self.at_ident_eq("rowid") {
                    return self.error("expected ROWID after WITHOUT");
                }
                self.bump();
                opts.without_rowid = true;
            } else if self.at_ident_eq("strict") {
                self.bump();
                opts.strict = true;
            } else {
                break;
            }
            if !self.eat(&TokenKind::Comma) {
                break;
            }
        }
        Ok(opts)
    }

    fn parse_column_def(&mut self) -> Result<ColumnDef> {
        let name = self.parse_ident()?;
        let type_name = if self.at_type_word() { Some(self.parse_type_name()?) } else { None };
        let mut constraints = Vec::new();
        while self.at_column_constraint_start() {
            constraints.push(self.parse_column_constraint()?);
        }
        Ok(ColumnDef { name, type_name, constraints })
    }

    fn at_column_constraint_start(&self) -> bool {
        matches!(
            self.kw(),
            Some(
                Keyword::Constraint
                    | Keyword::Primary
                    | Keyword::Not
                    | Keyword::Null
                    | Keyword::Unique
                    | Keyword::Check
                    | Keyword::Default
                    | Keyword::Collate
                    | Keyword::References
                    | Keyword::Generated
                    | Keyword::As
            )
        )
    }

    fn parse_column_constraint(&mut self) -> Result<ColumnConstraint> {
        let name = if self.eat_kw(Keyword::Constraint) { Some(self.parse_ident()?) } else { None };
        let kind = self.parse_column_constraint_kind()?;
        Ok(ColumnConstraint { name, kind })
    }

    fn parse_column_constraint_kind(&mut self) -> Result<ColumnConstraintKind> {
        match self.kw() {
            Some(Keyword::Primary) => {
                self.bump();
                self.expect_kw(Keyword::Key)?;
                let order = self.parse_opt_sort_order();
                let conflict = self.parse_opt_conflict_clause()?;
                let autoincrement = self.eat_kw(Keyword::Autoincrement);
                Ok(ColumnConstraintKind::PrimaryKey { order, conflict, autoincrement })
            }
            Some(Keyword::Not) => {
                self.bump();
                self.expect_kw(Keyword::Null)?;
                let conflict = self.parse_opt_conflict_clause()?;
                Ok(ColumnConstraintKind::NotNull { conflict })
            }
            Some(Keyword::Null) => {
                self.bump();
                let conflict = self.parse_opt_conflict_clause()?;
                Ok(ColumnConstraintKind::Null { conflict })
            }
            Some(Keyword::Unique) => {
                self.bump();
                let conflict = self.parse_opt_conflict_clause()?;
                Ok(ColumnConstraintKind::Unique { conflict })
            }
            Some(Keyword::Check) => {
                self.bump();
                self.expect(&TokenKind::LParen, "'(' after CHECK")?;
                let e = self.parse_expr()?;
                self.expect(&TokenKind::RParen, "')' after CHECK expression")?;
                Ok(ColumnConstraintKind::Check(e))
            }
            Some(Keyword::Default) => {
                self.bump();
                Ok(ColumnConstraintKind::Default(self.parse_default_value()?))
            }
            Some(Keyword::Collate) => {
                self.bump();
                Ok(ColumnConstraintKind::Collate(self.parse_ident()?))
            }
            Some(Keyword::References) => {
                Ok(ColumnConstraintKind::ForeignKey(self.parse_foreign_key_clause()?))
            }
            Some(Keyword::Generated) => {
                self.bump();
                self.expect_kw(Keyword::Always)?;
                self.expect_kw(Keyword::As)?;
                self.parse_generated_tail()
            }
            Some(Keyword::As) => {
                self.bump();
                self.parse_generated_tail()
            }
            _ => self.error("expected a column constraint"),
        }
    }

    fn parse_generated_tail(&mut self) -> Result<ColumnConstraintKind> {
        self.expect(&TokenKind::LParen, "'(' after GENERATED ... AS")?;
        let expr = self.parse_expr()?;
        self.expect(&TokenKind::RParen, "')' after generated column expression")?;
        let stored = if self.at_ident_eq("stored") {
            self.bump();
            true
        } else {
            self.eat_kw(Keyword::Virtual);
            false
        };
        Ok(ColumnConstraintKind::Generated { expr, stored })
    }

    fn parse_default_value(&mut self) -> Result<DefaultValue> {
        if self.eat(&TokenKind::LParen) {
            let first = self.pos;
            let e = self.parse_expr()?;
            // The expression's tokens are `first..self.pos`; keep their source text.
            let text = match self.pos.checked_sub(1) {
                Some(last) if last >= first => {
                    self.src[self.tokens[first].start..self.tokens[last].end].to_string()
                }
                _ => String::new(),
            };
            self.expect(&TokenKind::RParen, "')' after DEFAULT expression")?;
            return Ok(DefaultValue::Expr(Box::new(e), text));
        }
        let neg = if self.eat(&TokenKind::Minus) {
            true
        } else {
            self.eat(&TokenKind::Plus);
            false
        };
        match self.kind().clone() {
            TokenKind::Integer(v) => {
                self.bump();
                // `wrapping_neg`, not `-v`: a hex literal like `0x8000000000000000`
                // tokenizes to `i64::MIN`, and `-(i64::MIN)` overflows (a debug-build
                // panic). Wrapping folds `-0x8000000000000000` back to `i64::MIN`,
                // matching SQLite's two's-complement handling of the same literal.
                Ok(DefaultValue::Literal(Literal::Integer(if neg { v.wrapping_neg() } else { v })))
            }
            TokenKind::Real(v) => {
                self.bump();
                Ok(DefaultValue::Literal(Literal::Real(if neg { -v } else { v })))
            }
            _ if neg => self.error("expected a number after sign in DEFAULT"),
            TokenKind::Str(s) => {
                self.bump();
                Ok(DefaultValue::Literal(Literal::Text(s)))
            }
            // SQLite DQS legacy (quirks.html §8): a bare double-quoted DEFAULT token can
            // never resolve to a column (a column default has no row to read), so it is
            // UNCONDITIONALLY a text string literal — exactly the single-quoted case above.
            // Bracket/backtick quoting stays identifier-only and remains a parse error here
            // (falls through to the `_` arm), matching SQLite: only `"` gets the fallback.
            TokenKind::QuotedIdent { value, quote: QuoteKind::Double } => {
                self.bump();
                Ok(DefaultValue::Literal(Literal::Text(value)))
            }
            TokenKind::Blob(b) => {
                self.bump();
                Ok(DefaultValue::Literal(Literal::Blob(b)))
            }
            TokenKind::Keyword(Keyword::Null) => {
                self.bump();
                Ok(DefaultValue::Literal(Literal::Null))
            }
            TokenKind::Keyword(Keyword::CurrentDate) => {
                self.bump();
                Ok(DefaultValue::Literal(Literal::CurrentDate))
            }
            TokenKind::Keyword(Keyword::CurrentTime) => {
                self.bump();
                Ok(DefaultValue::Literal(Literal::CurrentTime))
            }
            TokenKind::Keyword(Keyword::CurrentTimestamp) => {
                self.bump();
                Ok(DefaultValue::Literal(Literal::CurrentTimestamp))
            }
            _ if self.at_ident_eq("true") => {
                self.bump();
                Ok(DefaultValue::Literal(Literal::True))
            }
            _ if self.at_ident_eq("false") => {
                self.bump();
                Ok(DefaultValue::Literal(Literal::False))
            }
            _ => self.error("expected a DEFAULT value"),
        }
    }

    fn parse_foreign_key_clause(&mut self) -> Result<ForeignKeyClause> {
        self.expect_kw(Keyword::References)?;
        let table = self.parse_ident()?;
        let columns =
            if self.check(&TokenKind::LParen) { self.parse_paren_name_list()? } else { Vec::new() };
        let mut actions = Vec::new();
        let mut match_name = None;
        loop {
            if self.eat_kw(Keyword::On) {
                let is_delete = if self.eat_kw(Keyword::Delete) {
                    true
                } else if self.eat_kw(Keyword::Update) {
                    false
                } else {
                    return self.error("expected DELETE or UPDATE after ON in foreign key");
                };
                let action = self.parse_referential_action()?;
                actions.push(if is_delete {
                    ForeignKeyAction::OnDelete(action)
                } else {
                    ForeignKeyAction::OnUpdate(action)
                });
            } else if self.eat_kw(Keyword::Match) {
                match_name = Some(self.parse_ident()?);
            } else {
                break;
            }
        }
        let deferrable = self.parse_opt_deferrable()?;
        Ok(ForeignKeyClause { table, columns, actions, match_name, deferrable })
    }

    fn parse_referential_action(&mut self) -> Result<ReferentialAction> {
        if self.eat_kw2(Keyword::Set, Keyword::Null) {
            Ok(ReferentialAction::SetNull)
        } else if self.eat_kw2(Keyword::Set, Keyword::Default) {
            Ok(ReferentialAction::SetDefault)
        } else if self.eat_kw(Keyword::Cascade) {
            Ok(ReferentialAction::Cascade)
        } else if self.eat_kw(Keyword::Restrict) {
            Ok(ReferentialAction::Restrict)
        } else if self.eat_kw2(Keyword::No, Keyword::Action) {
            Ok(ReferentialAction::NoAction)
        } else {
            self.error("expected SET NULL, SET DEFAULT, CASCADE, RESTRICT or NO ACTION")
        }
    }

    fn parse_opt_deferrable(&mut self) -> Result<Option<Deferrable>> {
        let (not, present) = if self.check_kw(Keyword::Deferrable) {
            (false, true)
        } else if self.check_kw(Keyword::Not) && self.kw_at(1) == Some(Keyword::Deferrable) {
            (true, true)
        } else {
            (false, false)
        };
        if !present {
            return Ok(None);
        }
        if not {
            self.bump();
        }
        self.expect_kw(Keyword::Deferrable)?;
        let initially = if self.eat_kw2(Keyword::Initially, Keyword::Deferred) {
            Some(InitiallyTiming::Deferred)
        } else if self.eat_kw2(Keyword::Initially, Keyword::Immediate) {
            Some(InitiallyTiming::Immediate)
        } else {
            None
        };
        Ok(Some(Deferrable { not, initially }))
    }

    fn parse_table_constraint(&mut self) -> Result<TableConstraint> {
        let name = if self.eat_kw(Keyword::Constraint) { Some(self.parse_ident()?) } else { None };
        let kind = match self.kw() {
            Some(Keyword::Primary) => {
                self.bump();
                self.expect_kw(Keyword::Key)?;
                let columns = self.parse_indexed_column_list()?;
                let conflict = self.parse_opt_conflict_clause()?;
                TableConstraintKind::PrimaryKey { columns, conflict }
            }
            Some(Keyword::Unique) => {
                self.bump();
                let columns = self.parse_indexed_column_list()?;
                let conflict = self.parse_opt_conflict_clause()?;
                TableConstraintKind::Unique { columns, conflict }
            }
            Some(Keyword::Check) => {
                self.bump();
                self.expect(&TokenKind::LParen, "'(' after CHECK")?;
                let e = self.parse_expr()?;
                self.expect(&TokenKind::RParen, "')' after CHECK expression")?;
                TableConstraintKind::Check(e)
            }
            Some(Keyword::Foreign) => {
                self.bump();
                self.expect_kw(Keyword::Key)?;
                let columns = self.parse_paren_name_list()?;
                let clause = self.parse_foreign_key_clause()?;
                TableConstraintKind::ForeignKey { columns, clause }
            }
            _ => return self.error("expected a table constraint"),
        };
        Ok(TableConstraint { name, kind })
    }

    pub(crate) fn parse_indexed_column_list(&mut self) -> Result<Vec<IndexedColumn>> {
        self.expect(&TokenKind::LParen, "'(' before indexed column list")?;
        let mut cols = vec![self.parse_indexed_column()?];
        while self.eat(&TokenKind::Comma) {
            cols.push(self.parse_indexed_column()?);
        }
        self.expect(&TokenKind::RParen, "')' after indexed column list")?;
        Ok(cols)
    }

    fn parse_indexed_column(&mut self) -> Result<IndexedColumn> {
        // `COLLATE` binds as an expression operator, so `a COLLATE x` lands as an
        // Expr(Collate) target rather than name+collation — semantically the same.
        let expr = self.parse_expr()?;
        let collation =
            if self.eat_kw(Keyword::Collate) { Some(self.parse_ident()?) } else { None };
        let order = self.parse_opt_sort_order();
        // `Expr` implements `Drop` (iterative teardown), so its fields can't be
        // moved out by pattern; match by reference and clone the bare column name
        // (cheap, and only for the plain-column case), else move the whole expr.
        let target = if let Expr::Column { schema: None, table: None, name, .. } = &expr {
            IndexedColumnTarget::Name(name.clone())
        } else {
            IndexedColumnTarget::Expr(expr)
        };
        Ok(IndexedColumn { target, collation, order })
    }

    // --- CREATE INDEX ---------------------------------------------------------

    fn parse_create_index(&mut self) -> Result<Statement> {
        let unique = self.eat_kw(Keyword::Unique);
        self.expect_kw(Keyword::Index)?;
        let if_not_exists = self.parse_if_not_exists()?;
        let name = self.parse_qualified_name()?;
        self.expect_kw(Keyword::On)?;
        let table = self.parse_ident()?;
        let columns = self.parse_indexed_column_list()?;
        let where_clause =
            if self.eat_kw(Keyword::Where) { Some(self.parse_expr()?) } else { None };
        Ok(Statement::CreateIndex(Box::new(CreateIndex {
            unique,
            if_not_exists,
            name,
            table,
            columns,
            where_clause,
        })))
    }

    // --- CREATE VIEW ----------------------------------------------------------

    fn parse_create_view(&mut self, temp: bool) -> Result<Statement> {
        self.expect_kw(Keyword::View)?;
        let if_not_exists = self.parse_if_not_exists()?;
        let name = self.parse_qualified_name()?;
        let columns =
            if self.check(&TokenKind::LParen) { Some(self.parse_paren_name_list()?) } else { None };
        self.expect_kw(Keyword::As)?;
        let select = self.parse_select()?;
        Ok(Statement::CreateView(Box::new(CreateView {
            temp,
            if_not_exists,
            name,
            columns,
            select: Box::new(select),
        })))
    }

    // --- CREATE TRIGGER -------------------------------------------------------

    fn parse_create_trigger(&mut self, temp: bool) -> Result<Statement> {
        self.expect_kw(Keyword::Trigger)?;
        let if_not_exists = self.parse_if_not_exists()?;
        let name = self.parse_qualified_name()?;
        let timing = if self.eat_kw(Keyword::Before) {
            Some(TriggerTiming::Before)
        } else if self.eat_kw(Keyword::After) {
            Some(TriggerTiming::After)
        } else if self.eat_kw2(Keyword::Instead, Keyword::Of) {
            Some(TriggerTiming::InsteadOf)
        } else {
            None
        };
        let event = if self.eat_kw(Keyword::Delete) {
            TriggerEvent::Delete
        } else if self.eat_kw(Keyword::Insert) {
            TriggerEvent::Insert
        } else if self.eat_kw(Keyword::Update) {
            let columns = if self.eat_kw(Keyword::Of) {
                let mut c = vec![self.parse_ident()?];
                while self.eat(&TokenKind::Comma) {
                    c.push(self.parse_ident()?);
                }
                c
            } else {
                Vec::new()
            };
            TriggerEvent::Update { columns }
        } else {
            return self.error("expected DELETE, INSERT or UPDATE in CREATE TRIGGER");
        };
        self.expect_kw(Keyword::On)?;
        // The ON-target may be schema-qualified (`ON main.tab`), the form the spec
        // recommends for a TEMP trigger on a non-temp table (lang_createtrigger.html §7);
        // a bare name (`ON tab`) parses as an unqualified `QualifiedName`.
        let table = self.parse_qualified_name()?;
        let for_each_row = if self.eat_kw(Keyword::For) {
            self.expect_kw(Keyword::Each)?;
            self.expect_kw(Keyword::Row)?;
            true
        } else {
            false
        };
        let when = if self.eat_kw(Keyword::When) { Some(self.parse_expr()?) } else { None };
        self.expect_kw(Keyword::Begin)?;
        let mut body = Vec::new();
        while !self.check_kw(Keyword::End) {
            if self.at_eof() {
                return self.error("unterminated CREATE TRIGGER body (expected END)");
            }
            body.push(self.parse_trigger_body_stmt()?);
            self.expect(&TokenKind::Semicolon, "';' after a trigger body statement")?;
        }
        self.expect_kw(Keyword::End)?;
        Ok(Statement::CreateTrigger(Box::new(CreateTrigger {
            temp,
            if_not_exists,
            name,
            timing,
            event,
            table,
            for_each_row,
            when,
            body,
        })))
    }

    /// A statement in a trigger body: INSERT / UPDATE / DELETE / SELECT only.
    fn parse_trigger_body_stmt(&mut self) -> Result<Statement> {
        match self.kw() {
            Some(Keyword::Insert) | Some(Keyword::Replace) => {
                Ok(Statement::Insert(Box::new(self.parse_insert(None)?)))
            }
            Some(Keyword::Update) => Ok(Statement::Update(Box::new(self.parse_update(None)?))),
            Some(Keyword::Delete) => Ok(Statement::Delete(Box::new(self.parse_delete(None)?))),
            Some(Keyword::Select) | Some(Keyword::With) | Some(Keyword::Values) => {
                Ok(Statement::Select(Box::new(self.parse_select()?)))
            }
            _ => self.error("trigger body statements must be INSERT, UPDATE, DELETE or SELECT"),
        }
    }

    // --- DROP -----------------------------------------------------------------

    pub(crate) fn parse_drop(&mut self) -> Result<Statement> {
        self.expect_kw(Keyword::Drop)?;
        let kind = if self.eat_kw(Keyword::Table) {
            DropKind::Table
        } else if self.eat_kw(Keyword::Index) {
            DropKind::Index
        } else if self.eat_kw(Keyword::View) {
            DropKind::View
        } else if self.eat_kw(Keyword::Trigger) {
            DropKind::Trigger
        } else {
            return self.error("expected TABLE, INDEX, VIEW or TRIGGER after DROP");
        };
        let if_exists = if self.eat_kw(Keyword::If) {
            self.expect_kw(Keyword::Exists)?;
            true
        } else {
            false
        };
        let name = self.parse_qualified_name()?;
        Ok(Statement::Drop(crate::ast_ddl::Drop { kind, if_exists, name }))
    }

    // --- ALTER TABLE ----------------------------------------------------------

    pub(crate) fn parse_alter_table(&mut self) -> Result<Statement> {
        self.expect_kw(Keyword::Alter)?;
        self.expect_kw(Keyword::Table)?;
        let name = self.parse_qualified_name()?;
        let action = if self.eat_kw(Keyword::Rename) {
            if self.eat_kw(Keyword::To) {
                AlterAction::RenameTo(self.parse_ident()?)
            } else {
                self.eat_kw(Keyword::Column);
                let from = self.parse_ident()?;
                self.expect_kw(Keyword::To)?;
                let to = self.parse_ident()?;
                AlterAction::RenameColumn { from, to }
            }
        } else if self.eat_kw(Keyword::Add) {
            self.eat_kw(Keyword::Column);
            AlterAction::AddColumn(self.parse_column_def()?)
        } else if self.eat_kw(Keyword::Drop) {
            self.eat_kw(Keyword::Column);
            AlterAction::DropColumn(self.parse_ident()?)
        } else {
            return self.error("expected RENAME, ADD or DROP after ALTER TABLE name");
        };
        Ok(Statement::AlterTable(Box::new(AlterTable { name, action })))
    }

    // --- shared constraint helpers (also used by DML) -------------------------

    pub(crate) fn parse_opt_sort_order(&mut self) -> Option<SortOrder> {
        if self.eat_kw(Keyword::Asc) {
            Some(SortOrder::Asc)
        } else if self.eat_kw(Keyword::Desc) {
            Some(SortOrder::Desc)
        } else {
            None
        }
    }

    pub(crate) fn parse_opt_conflict_clause(&mut self) -> Result<Option<ConflictClause>> {
        if self.eat_kw2(Keyword::On, Keyword::Conflict) {
            Ok(Some(self.parse_conflict_algorithm()?))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn parse_conflict_algorithm(&mut self) -> Result<ConflictClause> {
        if self.eat_kw(Keyword::Rollback) {
            Ok(ConflictClause::Rollback)
        } else if self.eat_kw(Keyword::Abort) {
            Ok(ConflictClause::Abort)
        } else if self.eat_kw(Keyword::Fail) {
            Ok(ConflictClause::Fail)
        } else if self.eat_kw(Keyword::Ignore) {
            Ok(ConflictClause::Ignore)
        } else if self.eat_kw(Keyword::Replace) {
            Ok(ConflictClause::Replace)
        } else {
            self.error("expected ROLLBACK, ABORT, FAIL, IGNORE or REPLACE")
        }
    }
}
