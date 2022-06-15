grammar toy;

module 
    : structdefine* fundefinition+
    ;


expression
    : expression (Dian) expression 
    | expression  (MulDiv) expression
    | expression  (AddSub) expression
    | Number
    | tensorLiteral
    | Parenthese_open expression Parenthese_close
    | structLiteral
    | identifierexpr
    ; 


identifierexpr
    : Identifier
    | Identifier Parenthese_open (expression(Comma expression)*)? Parenthese_close
    ;


returnExpression 
    : Return 
    | Return expression 
    ; 

tensorLiteral
    : Sbracket_open (tensorLiteral (Comma tensorLiteral)*)? Sbracket_close 
    | Number 
    ;

literalList
    : tensorLiteral
    | tensorLiteral Comma literalList
    ;

structLiteral
    : Bracket_open (literalList)+ Bracket_close
    ;  

decl
    : Var Identifier  (type)?  (Equal expression)?
    | Identifier Identifier (Equal expression)?
    ;

type
    : AngleBrackets_open (Number (Comma Number)*)? AngleBrackets_close
    ;


fundefinition
    : prototype block
    ;

prototype
    : Def Identifier Parenthese_open decl_list Parenthese_close
    ;

decl_list 
    : Identifier 
    | Identifier Comma decl_list
    |
    ;

block
    : Bracket_open (block_expr Semicolon)* Bracket_close
    ;

block_expr
    : decl | returnExpression | expression 
    ;

structdefine
    : 'struct' Identifier Bracket_open (decl Semicolon)* Bracket_close
    ;

Parenthese_open 
    :'('
    ;
Parenthese_close 
    : ')'
    ;
Bracket_open 
    :'{'
    ;

Bracket_close 
    : '}'
    ;
Sbracket_open 
    :'['
    ;
Sbracket_close 
    :']'
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
    : [0-9] ([0-9])*
    ;



Equal
    : '='
    ;

AngleBrackets_open 
    : '<'
    ;
AngleBrackets_close
    : '>' 
    ;
Comma
    : ','
    ;

WS
    : [ \r\n\t] -> skip
    ;

AddSub
    : '+'
    | '-'
    ;

MulDiv
    : '*'
    | '/'
    ;

Dian
    : '.'
    ;

Comment 
    : '#' .*? '\n' ->skip
    ;
