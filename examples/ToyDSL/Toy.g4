grammar Toy;

@parser::members {
  std::vector<double> tensorDataBuffer;
}

module 
    : structDefine* funDefine+
    ;

expression
    : Number
    | tensorLiteral 
      {
        tensorDataBuffer.clear();
      }
    | identifierExpr
    | expression Mul expression
    | expression Add expression
    | expression Dot expression
    | structLiteral
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
    | Identifier Identifier (Equal expression)?
      {
        $idName = $Identifier.text;
      }
    | Identifier 
      {
        $idName = $Identifier.text;
      }
    ;

type
    : AngleBracketsOpen Number (Comma Number)* AngleBracketsClose
    ;

funDefine
    : prototype block
    ;

prototype returns [std::string idName]
    : Def Identifier ParentheseOpen declList? ParentheseClose
      {
        $idName = $Identifier.text;
      }
    ;

declList 
    : varDecl
    | varDecl Comma declList
    ;

block
    : BracketOpen (blockExpr Semicolon)* BracketClose
    ;

blockExpr
    : varDecl | returnExpr | expression 
    ;

literalList
    : tensorLiteral
      {
        tensorDataBuffer.clear();
      } 
    | tensorLiteral Comma literalList
    ;

structLiteral 
    :  
    | BracketOpen (structLiteral | literalList) (Comma (structLiteral| literalList))* BracketClose    
    ;

structDefine
    : Struct Identifier BracketOpen (varDecl Semicolon)* BracketClose 
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

Struct 
    : 'struct'
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

Add  
    : '+'
    ;

Mul 
    : '*'
    ;

Dot 
    : '.'
    ;

WS
    : [ \r\n\t] -> skip
    ;
    
Comment 
    : '#' .*? '\n' ->skip
    ;
