grammar Toy;

module
  : funDefine 
  ;

expression
  : Number 
  | tensorLiteral 
  | identifierExpr 
  | expression Add expression 
  ;

returnExpr
  : Return expression ?
  ;

identifierExpr
  : Identifier 
  | Identifier ParentheseOpen (expression (Comma expression )*)?ParentheseClose 
  ;

tensorLiteral
  : SbracketOpen (tensorLiteral (Comma tensorLiteral )*)?SbracketClose 
  | Number 
  ;

varDecl returns [std::string idName]
  : Var Identifier (type)? (Equal expression)?
    {
    // Record the identifier string to `idName` member.
    $idName = $Identifier.text;
    }
  ;

type
  : AngleBracketOpen Number (Comma Number )*AngleBracketClose 
  ;

funDefine
  : prototype block 
  ;

prototype
  : Def Identifier ParentheseOpen declList ?ParentheseClose 
  ;

declList
  : Identifier 
  | Identifier Comma declList 
  ;

block
  : BracketOpen (blockExpr Semi )*BracketClose 
  ;

blockExpr
  : varDecl 
  | returnExpr 
  | expression 
  ;

Return
  : 'return'
  ;

ParentheseOpen
  : '('
  ;

SbracketOpen
  : '['
  ;

ParentheseClose
  : ')'
  ;

SbracketClose
  : ']'
  ;

Var
  : 'var'
  ;

Add
  : 'add'
  ;

Sub
  : 'sub'
  ;

Number
  : [0-9]+
  ;

Comma
  : ','
  ;

Semi
  : ';'
  ;

BracketOpen
  : '{'
  ;

Def
  : 'def'
  ;

BracketClose
  : '}'
  ;

AngleBracketOpen
  : '<'
  ;

Equal
  : '='
  ;

AngleBracketClose
  : '>'
  ;

Identifier
  : [a-zA-Z][a-zA-Z0-9_]*
  ;

WS
  : [ \r\n\t] -> skip
  ;

Comment
  : '#' .*? '\n' ->skip
  ;
