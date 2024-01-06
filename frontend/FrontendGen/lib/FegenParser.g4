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
	: LeftBracket valueSpec (Comma valueSpec)* RightBracket
	;

valueSpec
	: tdValueSpec
	| cppValueSpec
	;

tdValueSpec
	: tdValueKind Less tdTypeSpec (Comma tdTypeSpec)* Greater identifier?
	;

tdValueKind
	: OPERAND_VALUE
	| ATTRIBUTE_VALUE
	;

cppValueSpec
	: CPP_VALUE Less cppTypeSpec (Comma cppTypeSpec)* Greater identifier?
	;

tdTypeSpec
	: builtinType
	| userDefineType
	;

builtinType
	: INT
	| FLOAT
	| TENSOR
	;

userDefineType
	: identifier (Dot identifier)?
	;

cppTypeSpec
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
	: irKind Less tdTypeSpec Greater
	;

irKind
	: OP_IR
	| ATTRIBUTE_IR
	| TYPE_IR
	;

identifier
    : LexerRuleName
    | ParserRuleName
    ;