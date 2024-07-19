parser grammar FegenParser;

options {
    tokenVocab = FegenLexer;
}

fegenSpec
    : fegenDecl (prequelConstruct | functionDecl | typeDefinationDecl | statement | opDecl | rules)* EOF
    ;

fegenDecl
    : FEGEN identifier
    ;

// preprocess declare
prequelConstruct
    : BeginInclude INCLUDE_CONTENT* EndInclude
    ;

// function declare
functionDecl
    : typeSpec funcName LeftParen funcParams? RightParen statementBlock
    ;

funcName
    : identifier
    ;

funcParams
    : typeSpec identifier (Comma typeSpec identifier)*
    ;

// typedef declare
typeDefinationDecl
    : TYPEDEF typeDefinationName typeDefinationBlock
    ; 

typeDefinationName
    : identifier
    ;

typeDefinationBlock
    : LeftBrace parametersSpec assemblyFormatSpec? RightBrace 
    ;

parametersSpec
    : PARAMETERS varDecls
    ;

assemblyFormatSpec
    : ASSEMBLY_FORMAT LeftBracket StringLiteral RightBracket
    ;

// opdef declare
opDecl
    : OPDEF opName opBlock
    ;

opName
    : identifier
    ;

opBlock
    : LeftBrace argumentSpec? resultSpec? bodySpec? RightBrace
    ;

argumentSpec
    : ARGUMENTS varDecls
    ;

resultSpec
    : RESULTS varDecls
    ;

bodySpec
    : BODY statementBlock
    ;

// rule definations
rules
    : ruleSpec+
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

// action block declare
actionBlock
    : LeftBrace inputsSpec? returnsSpec? actionSpec? RightBrace
    ;

inputsSpec
	: INPUTS varDecls
	;

varDecls
	: LeftBracket typeSpec identifier (Comma typeSpec identifier)* RightBracket
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
    : ACTIONS statementBlock
    ;

statementBlock
    : LeftBrace statement* RightBrace
    ;

statement
    : varDeclStmt Semi
    | assignStmt Semi
    | functionCall Semi
    | opInvokeStmt Semi
    | ifStmt
    | forStmt
    | returnBlock Semi
    ;

varDeclStmt
    : typeSpec identifier (Assign expression)?
    ;

assignStmt
    : identifier Assign expression
    ;

functionCall
    : funcName LeftParen (expression (Comma expression)*)? RightParen
    ;

opInvokeStmt
    : opName LeftParen opParams? (Comma opResTypeParams)? RightParen+
    ;

opParams
    : identifier (Comma identifier)*
    ;

opResTypeParams
    : typeInstance (Comma typeInstance)*
    ;

ifStmt
    :  ifBlock (ELSE ifBlock)* (elseBlock)?
    ;

ifBlock:
    IF LeftParen expression RightParen statementBlock
    ;

elseBlock
    : ELSE statementBlock
    ;

forStmt
    : FOR LeftParen (assignStmt | varDeclStmt) Semi expression Semi assignStmt RightParen statementBlock
    ;

returnBlock
    : RETURN expression
    ;

// expression
expression
    : andExpr (Logic_OR andExpr)*
    ;

andExpr
    : equExpr  (AND equExpr )*
    ;

equExpr 
    : compareExpr  ((EQUAL | NOT_EQUAL) compareExpr)*
    ;

compareExpr
    : addExpr ((Less | LessEqual | Greater | GreaterEqual) addExpr)*
    ;

addExpr
    : term ((Plus | Minus) term)*
    ;

term
    : powerExpr ((Star | Div | MOD) powerExpr)*
    ;

powerExpr
    : unaryExpr (StarStar unaryExpr)*
    ;

unaryExpr
    : (Minus | Plus | Exclamation)? primaryExpr
    ;

parenSurroundedExpr
    : LeftParen expression RightParen
    ;

primaryExpr
    : constant
    | identifier
    | functionCall
    | parenSurroundedExpr
    | contextMethodInvoke
    | typeSpec
    | variableAccess
    ;

constant
    : numericLiteral
    | charLiteral
    | boolLiteral
    | listLiteral
    ;

// ex: $ctx(0).getText()
contextMethodInvoke
    : Dollar identifier LeftParen intLiteral? RightParen Dot functionCall
    ;

variableAccess
    : identifier LeftBracket expression RightBracket
    ;

numericLiteral
    : intLiteral
    | realLiteral
    ;

intLiteral
    : UnsignedInt
    | (Plus | Minus) UnsignedInt
    ;

realLiteral
    : ScienceReal
    ;

charLiteral
    : StringLiteral
    ;

boolLiteral
    : ConstBoolean
    ;

listLiteral
    : LeftBracket (expression (Comma expression)*)? RightBracket
    ;

// type system
typeSpec
    : valueKind? typeInstance # typeInstanceSpec
    | valueKind? typeTemplate # typeTemplateSpce
    | valueKind? collectType # collectTypeSpec
    ;

valueKind
    : CPP
    | OPERAND
    | ATTRIBUTE
    ;


typeInstance
    : typeTemplate Less typeTemplateParam (Comma typeTemplateParam)* Greater
    | builtinTypeInstances
    | identifier
    ;

typeTemplate
    : prefixedName
    | builtinTypeTemplate
    | TYPE
    ;

typeTemplateParam
    : expression
    | builtinTypeInstances
    ;

builtinTypeInstances
    : BOOL
    | INT
    | FLOAT
    | DOUBLE
    | CHAR
    | STRING
    ;

builtinTypeTemplate
    : INTEGER
    | FLOATPOINT
    | TENSOR
    | VECTOR
    ;

collectType
    : collectProtoType Less expression Greater
    ;

collectProtoType
    : ANY
    | LIST
    | OPTIONAL
    ;