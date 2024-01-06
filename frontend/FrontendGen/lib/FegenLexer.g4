lexer grammar FegenLexer;

fragment Schar: ~ ["\\\r\n];

fragment NONDIGIT: [a-zA-Z_];

fragment UPPERCASE: [A-Z];

fragment LOWERCASE: [a-z];

fragment DIGIT: [0-9];

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

LexerRuleName: UPPERCASE (NONDIGIT | DIGIT)*;

ParserRuleName: LOWERCASE (NONDIGIT | DIGIT)*;

StringLiteral: '\'' Schar* '\'';

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

Colon : ':';

OR : '|';

QuestionMark : '?';

Star : '*';

Plus : '+';

Whitespace: [ \t]+ -> skip;

Newline: ('\r' '\n'? | '\n') -> skip;

BlockComment: '/*' .*? '*/' -> skip;

LineComment: '//' ~ [\r\n]* -> skip;