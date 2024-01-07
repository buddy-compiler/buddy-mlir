parser grammar FegenParser;

options {
    tokenVocab = FegenLexer;
}

fegenModule
	: (parserGrammarNode | lexerGrammarNode)+
	;

//TODO: add lexer grammar definaion to reduce sema work
lexerGrammarNode
    : LexerRuleName grammarSpec
    ;

parserGrammarNode
	: ParserRuleName inputsSpec? returnsSpec? grammarSpec irSpec?
	;


inputsSpec
	: INPUTS valueDecls
	;

returnsSpec
	: RETURNS valueDecls cppCode?
	;

valueDecls
	: LeftBracket valueDeclSpec (Comma valueDeclSpec)* RightBracket
	;

valueDeclSpec
	: listValueDeclSpec
    | tdValueDeclSpec
	| cppValueDeclSpec
	;

listValueDeclSpec
    : LIST Less (tdTypeSpec | cppTypeSpec) Greater identifier?
    ;

tdValueDeclSpec
	: tdTypeSpec (identifier (Assign tdValueSpec)?)?
	;

tdTypeSpec
    : tdValueKind Less tdType (Comma tdType)* Greater
    ;

tdValueSpec
    : attrRef
    | StringLiteral
    | tensorLiteral
    | expression
    ;

attrRef
    : Dollar identifier Dot identifier
    ;

tensorLiteral
    : LeftBracket (tensorLiteral (Comma tensorLiteral)*)? RightBracket
    | SignedIntLiteral
    | RealLiteral
    ;

expression
    : term ( ( Plus | Minus ) term )*
    ;

term
    : powerExpr ( (Star | Div | MOD) powerExpr )*
    ;

powerExpr
    : unaryExpr ( StarStar unaryExpr )*
    ;

unaryExpr
    : (Minus | Plus) ? primaryExpression
    ;

primaryExpression
    : SignedIntLiteral
    | RealLiteral
    | parenSurroundedExpr
    ;

parenSurroundedExpr
    : LeftParen expression RightParen
    ;

tdType
	: builtinType
	| userDefineType
	;

tdValueKind
	: OPERAND_VALUE
	| ATTRIBUTE_VALUE
	;

cppValueDeclSpec
	: cppTypeSpec identifier?
	;

cppTypeSpec
    : CPP_VALUE Less cppType (Comma cppType)* Greater
    ;

builtinType
	: INT
	| FLOAT
	| TENSOR
    | STRING
	;

userDefineType
	: identifier (Dot identifier)?
	;

cppType
	: StringLiteral
	;

// TODO:  push to cpp channel
cppCode
	: LeftBrace RightBrace
	;

grammarSpec
	: GRAMMAR LeftBracket antlrRule RightBracket
	;

antlrRule
	: alternatives+
	;

alternatives
    : alternative ('|' alternative)*
    ;

alternative
    : suffixedRule ruleSuffix?
    ;

suffixedRule
    : parenSurroundedElem
    | StringLiteral
    | identifier
    ;

parenSurroundedElem
    : LeftParen antlrRule RightParen
    ;

ruleSuffix
    : Star QuestionMark
    | Plus QuestionMark
    | QuestionMark
    | Star
    | Plus
    ;

irSpec
	: IR LeftBracket singleIrDecl (Comma singleIrDecl)*  RightBracket
	;

singleIrDecl
	: irKind Less tdType Greater irParameters?
	;

irKind
	: OP_IR
	| ATTRIBUTE_IR
	| TYPE_IR
	;

irParameters
    : LeftParen irInputs* Arror irOutputs* RightParen
    ;

irInputs
    : identifier
    ;

irOutputs
    : identifier
    ;

identifier
    : LexerRuleName
    | ParserRuleName
    ;