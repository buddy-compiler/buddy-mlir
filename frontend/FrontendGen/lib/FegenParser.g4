parser grammar FegenParser;

options {
    tokenVocab = FegenLexer;
}

fegenSpec
    : fegenDecl rules EOF
    ;

fegenDecl
    : FEGEN identifier Semi
    ;

rules
    : ruleSpec*
    ;

ruleSpec
    : parserRuleSpec
    | lexerRuleSpec
    ;

parserRuleSpec
    : ParserRuleName Colon ruleBlock Semi
    ;

ruleBlock
    : ruleAltList
    ;

ruleAltList
    : actionAlt (OR actionAlt)*
    ;

actionAlt
    : alternative actionBlock?
    ;

alternative
    : element*
    ;

element
    : atom (ebnfSuffix |)
    | ebnf
    ;

atom
    : terminalDef
    | ruleref
    | notSet
    ;

// terminal rule reference
terminalDef
    : LexerRuleName
    | StringLiteral
    ;

// parser rule reference
ruleref
    : ParserRuleName
    ;

notSet
    : Tilde setElement
    | Tilde blockSet
    ;

setElement
    : LexerRuleName
    | StringLiteral
    | characterRange
    ;

characterRange
    : StringLiteral Range StringLiteral
    ;

blockSet
    : LeftParen setElement (OR setElement)* RightParen
    ;

ebnfSuffix
    : QuestionMark QuestionMark?
    | Star QuestionMark?
    | Plus QuestionMark?
    ;

ebnf
    : block blockSuffix?
    ;

block
    : LeftParen altList RightParen
    ;

blockSuffix
    : ebnfSuffix
    ;

altList
    : alternative (OR alternative)*
    ;

// lexer rule
lexerRuleSpec
    : LexerRuleName Colon lexerRuleBlock Semi
    ;

lexerRuleBlock
    : lexerAltList
    ;

lexerAltList
    : lexerAlt (OR lexerAlt)*
    ;

lexerAlt
    : lexerElements lexerCommands?
    |
    ;

// E.g., channel(HIDDEN), skip, more, mode(INSIDE), push(INSIDE), pop

lexerCommands
    : Arror lexerCommand (Comma lexerCommand)*
    ;

lexerCommand
    : lexerCommandName
    ;

lexerCommandName
    : identifier
    ;

lexerElements
    : lexerElement+
    |
    ;

lexerElement
    : lexerAtom ebnfSuffix?
    | lexerBlock ebnfSuffix?
    ;

lexerAtom
    : characterRange
    | terminalDef
    | notSet
    | Dot
    ;

lexerBlock
    : LeftParen lexerAltList RightParen
    ;

actionBlock
    : LeftBrace inputsSpec? returnsSpec? actionSpec? irSpec? RightBrace
    ;

inputsSpec
	: INPUTS varDecls
	;

varDecls
	: LeftBracket varDeclSpec (Comma varDeclSpec)* RightBracket
	;

varDeclSpec
	: type identifier?
	;

type
    : LIST Less type Greater
    | CPP_VALUE Less StringLiteral Greater
    | (OPERAND_VALUE | ATTRIBUTE_VALUE) Less prefixedName (Comma prefixedName)* Greater
    ;

prefixedName
    : identifier (Dot identifier)? 
    ;

identifier
    : LexerRuleName
    | ParserRuleName
    ;

returnsSpec
	: RETURNS varDecls
	;

actionSpec
    : ACTIONS LeftBrace (statement Semi)* RightBrace
    ;

statement
    : varDeclStmt
    | functionCallStmt
    | assignStmt
    ;

varDeclStmt
    : type identifier (Assign (variable | functionCallStmt))?
    ;

functionCallStmt
    :  functionAccess LeftParen paramList? RightParen
    ;

functionAccess
    : FUNCTION Less StringLiteral Greater
    | (OPERATION | ATTRIBUTE) Less identifier (Dot identifier)? Greater
    | builtinFunction
    ;

builtinFunction
    : identifier
    ;


paramList
    : variable (Comma variable)*
    ;

assignStmt
    : identifier Assign expression
    ;

expression
    : variable
    | functionCallStmt
    | NULL
    ;

variable
    : identifier
    | Dollar identifier LeftParen IntLiteral? RightParen ruleSuffix?
    ;

ruleSuffix
    : Dot variable
    | Dot RETURNS LeftBracket IntLiteral RightBracket
    ;   

irSpec
    : IR LeftBracket (OPERATION | ATTRIBUTE) Less identifier (Dot identifier)? Greater RightBracket
    ;