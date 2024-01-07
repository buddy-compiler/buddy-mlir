lexer grammar FegenLexer;

fragment Schar: ~ ["\\\r\n];

fragment NONDIGIT: [a-zA-Z_];

fragment UPPERCASE: [A-Z];

fragment LOWERCASE: [a-z];

fragment NOZERODIGIT: [1-9];

fragment DIGIT: [0-9];

// key words

INPUTS: 'inputs';

RETURNS: 'returns';

GRAMMAR: 'grammar';

IR: 'ir';

OPERAND_VALUE: 'operandValue';

ATTRIBUTE_VALUE: 'attributeValue';

CPP_VALUE: 'cppValue';

OP_IR: 'operation';

ATTRIBUTE_IR: 'attribute';

TYPE_IR: 'type';

INT: 'int';

FLOAT: 'float';

TENSOR: 'tensor';

// identifiers

LexerRuleName: UPPERCASE (NONDIGIT | DIGIT)*;

ParserRuleName: LOWERCASE (NONDIGIT | DIGIT)*;

// literal

StringLiteral: '\'' Schar* '\'';

SignedIntLiteral: (Plus | Minus)? UnsignedIntLiteral;

UnsignedIntLiteral: '0' | NOZERODIGIT DIGIT*;

RealLiteral: SignedIntLiteral Dot UnsignedIntLiteral;

// marks

Less: '<';

Greater: '>';

Comma: ',';

Semi: ';';

LeftParen: '(';

RightParen: ')';

LeftBracket: '[';

RightBracket: ']';

LeftBrace: '{';

RightBrace: '}';

Dot: '.';

Colon: ':';

OR: '|';

QuestionMark: '?';

Star: '*';

Div: '/';

Plus: '+';

Minus: '-';

Assign: '=';

Dollar: '$';

StarStar: '**';

Whitespace: [ \t]+ -> skip;

Newline: ('\r' '\n'? | '\n') -> skip;

BlockComment: '/*' .*? '*/' -> skip;

LineComment: '//' ~ [\r\n]* -> skip;