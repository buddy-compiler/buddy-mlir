grammar Toy;

@parser::members {
  std::vector<double> tensorDataBuffer;
}

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

tensorLiteral returns [std::vector<double> data]
    : SbracketOpen (tensorLiteral (Comma tensorLiteral)*)? SbracketClose 
      {
        // When the `]` is detected, copy the elements of `tensorDataBuffer` to `data` member.
        // Suppose we are handling the `[[1, 2], [3, 4]]` layout.
        // - The `[1, 2]` will be insert to `tensorDataBuffer`.
        // - The elements of `tensorDataBuffer` will be assign to `data` member (1, 2).
        // - The `[3, 4]` will be insert to `tensorDataBuffer` (1, 2, 3, 4).
        // - The elements of `tensorDataBuffer` will be assign to `data` member (1, 2, 3, 4).
        if ($SbracketClose) 
          $data.assign(tensorDataBuffer.begin(), tensorDataBuffer.end());
      }
    | Number 
      {
        // Insert current data into `tensorDataBuffer`.
        tensorDataBuffer.push_back((double)$Number.int); 
      }
    ;

varDecl returns [std::string idName]
    : Var Identifier (type)? (Equal expression)?
      {
        // Record the identifier string to `idName` member.
        $idName = $Identifier.text;
        // Clear the `tensorDataBuffer` before accessing `tensorLiteral`.
        if ($Equal)
          tensorDataBuffer.clear();
      }
    ;

type
    : AngleBracketsOpen Number (Comma Number)* AngleBracketsClose
    ;

funDefine
    : prototype block
    ;

prototype
    : Def Identifier ParentheseOpen declList ParentheseClose
    ;

declList 
    : Identifier
    | Identifier Comma declList
    |
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
