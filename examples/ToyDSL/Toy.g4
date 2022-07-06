grammar Toy;

module 
    : funDefine+
    ;

expression
    : Number
    | tensorLiteral
    | identifierExpr
    ; 

identifierExpr
    : Identifier
    | Identifier ParentheseOpen (expression(Comma expression)*)? ParentheseClose
    ;

returnExpr
    : Return 
    | Return expression 
    ; 

tensorLiteral
    : SbracketOpen (tensorLiteral (Comma tensorLiteral)*)? SbracketClose 
    | Number 
    ;

varDecl
    : Var Identifier (type)? (Equal expression)?
    ;

type
    : AngleBracketsOpen Number (Comma Number)* AngleBracketsClose
    ;

funDefine
    : prototype block
    ;

prototype
    : Def Identifier ParentheseOpen declList? ParentheseClose
    ;

declList 
    : Identifier 
    | Identifier Comma declList
    ;

block
    : BracketOpen (blockExpr Semicolon)* BracketClose
    ;

blockExpr
    : varDecl | returnExpr | expression 
    ;

ParentheseOpen 
    : '('
    ;

ParentheseClose 
    : ')'
    ;

BracketOpen 
    : '{'
    ;

BracketClose 
    : '}'
    ;

SbracketOpen 
    : '['
    ;

SbracketClose 
    : ']'
    ;

Return
    : 'return'
    ;
    
Semicolon
    : ';'
    ;

Var 
    : 'var'
    ;

Def 
    : 'def'
    ;

Identifier
    : [a-zA-Z][a-zA-Z0-9_]*
    ;

Number
    : [0-9]+
    ;

Equal
    : '='
    ;

AngleBracketsOpen 
    : '<'
    ;

AngleBracketsClose
    : '>' 
    ;

Comma
    : ','
    ;

WS
    : [ \r\n\t] -> skip
    ;
    
Comment 
    : '#' .*? '\n' ->skip
    ;
