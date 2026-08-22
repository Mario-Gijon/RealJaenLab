const PITCH_MODEL = {
  length: 105,
  width: 68,
  centerCircleRadius: 9.15,
  landmarks: [
    ["top_left_corner", "Esquina superior izquierda", -52.5, -34],
    ["top_right_corner", "Esquina superior derecha", 52.5, -34],
    ["bottom_right_corner", "Esquina inferior derecha", 52.5, 34],
    ["bottom_left_corner", "Esquina inferior izquierda", -52.5, 34],
    ["halfway_top", "Mitad de campo / banda superior", 0, -34],
    ["halfway_bottom", "Mitad de campo / banda inferior", 0, 34],
    ["centre_mark", "Punto central", 0, 0],
    ["left_penalty_top", "Esquina superior del área izquierda", -36, -20.16],
    ["left_penalty_bottom", "Esquina inferior del área izquierda", -36, 20.16],
    ["right_penalty_top", "Esquina superior del área derecha", 36, -20.16],
    ["right_penalty_bottom", "Esquina inferior del área derecha", 36, 20.16],
    ["left_goal_top", "Esquina superior del área pequeña izquierda", -47, -9.16],
    ["left_goal_bottom", "Esquina inferior del área pequeña izquierda", -47, 9.16],
    ["right_goal_top", "Esquina superior del área pequeña derecha", 47, -9.16],
    ["right_goal_bottom", "Esquina inferior del área pequeña derecha", 47, 9.16],
    ["left_penalty_spot", "Punto de penalti izquierdo", -41.5, 0],
    ["right_penalty_spot", "Punto de penalti derecho", 41.5, 0],
    ["left_goal_post_top", "Base del poste superior de la portería izquierda (arriba en el diagrama)", -52.5, -3.66],
    ["left_goal_post_bottom", "Base del poste inferior de la portería izquierda (abajo en el diagrama)", -52.5, 3.66],
    ["right_goal_post_top", "Base del poste superior de la portería derecha (arriba en el diagrama)", 52.5, -3.66],
    ["right_goal_post_bottom", "Base del poste inferior de la portería derecha (abajo en el diagrama)", 52.5, 3.66],
    ["left_penalty_goal_line_top", "Intersección área de penalti / línea de gol izquierda, superior (diagrama)", -52.5, -20.16],
    ["left_penalty_goal_line_bottom", "Intersección área de penalti / línea de gol izquierda, inferior (diagrama)", -52.5, 20.16],
    ["right_penalty_goal_line_top", "Intersección área de penalti / línea de gol derecha, superior (diagrama)", 52.5, -20.16],
    ["right_penalty_goal_line_bottom", "Intersección área de penalti / línea de gol derecha, inferior (diagrama)", 52.5, 20.16],
    ["left_goal_area_goal_line_top", "Intersección área pequeña / línea de gol izquierda, superior (diagrama)", -52.5, -9.16],
    ["left_goal_area_goal_line_bottom", "Intersección área pequeña / línea de gol izquierda, inferior (diagrama)", -52.5, 9.16],
    ["right_goal_area_goal_line_top", "Intersección área pequeña / línea de gol derecha, superior (diagrama)", 52.5, -9.16],
    ["right_goal_area_goal_line_bottom", "Intersección área pequeña / línea de gol derecha, inferior (diagrama)", 52.5, 9.16],
    ["centre_circle_halfway_top", "Intersección círculo central / medio campo, superior (diagrama)", 0, -9.15],
    ["centre_circle_halfway_bottom", "Intersección círculo central / medio campo, inferior (diagrama)", 0, 9.15],
  ].map(([id, label, x, y]) => ({ id, label, world: { x, y } })),
};

PITCH_MODEL.landmarksById = Object.fromEntries(PITCH_MODEL.landmarks.map((landmark) => [landmark.id, landmark]));

PITCH_MODEL.calibrationLines = [
  ["halfway_line", "Línea de medio campo", [0, -34], [0, 34]],
  ["top_touchline", "Banda superior", [-52.5, -34], [52.5, -34]],
  ["bottom_touchline", "Banda inferior", [-52.5, 34], [52.5, 34]],
  ["left_goal_line", "Línea de fondo izquierda", [-52.5, -34], [-52.5, 34]],
  ["right_goal_line", "Línea de fondo derecha", [52.5, -34], [52.5, 34]],
  ["left_penalty_top", "Área izquierda — línea superior", [-52.5, -20.16], [-36, -20.16]],
  ["left_penalty_front", "Área izquierda — línea frontal", [-36, -20.16], [-36, 20.16]],
  ["left_penalty_bottom", "Área izquierda — línea inferior", [-36, 20.16], [-52.5, 20.16]],
  ["right_penalty_top", "Área derecha — línea superior", [52.5, -20.16], [36, -20.16]],
  ["right_penalty_front", "Área derecha — línea frontal", [36, -20.16], [36, 20.16]],
  ["right_penalty_bottom", "Área derecha — línea inferior", [36, 20.16], [52.5, 20.16]],
  ["left_goal_top", "Área pequeña izquierda — línea superior", [-52.5, -9.16], [-47, -9.16]],
  ["left_goal_front", "Área pequeña izquierda — línea frontal", [-47, -9.16], [-47, 9.16]],
  ["left_goal_bottom", "Área pequeña izquierda — línea inferior", [-47, 9.16], [-52.5, 9.16]],
  ["right_goal_top", "Área pequeña derecha — línea superior", [52.5, -9.16], [47, -9.16]],
  ["right_goal_front", "Área pequeña derecha — línea frontal", [47, -9.16], [47, 9.16]],
  ["right_goal_bottom", "Área pequeña derecha — línea inferior", [47, 9.16], [52.5, 9.16]],
].map(([id, label, start, end]) => ({ id, label, world: { start: { x: start[0], y: start[1] }, end: { x: end[0], y: end[1] } } }));
PITCH_MODEL.calibrationLinesById = Object.fromEntries(PITCH_MODEL.calibrationLines.map((line) => [line.id, line]));
PITCH_MODEL.penaltyArcAngle = Math.acos(5.5 / PITCH_MODEL.centerCircleRadius);
PITCH_MODEL.calibrationArcs = [
  {
    id: "centre_circle",
    label: "Círculo central",
    world: { center: { x: 0, y: 0 }, radius: PITCH_MODEL.centerCircleRadius },
  },
  {
    id: "left_penalty_arc",
    label: "Arco de penalti izquierdo (exterior del área)",
    world: {
      center: { x: -41.5, y: 0 }, radius: PITCH_MODEL.centerCircleRadius,
      startAngle: -PITCH_MODEL.penaltyArcAngle, endAngle: PITCH_MODEL.penaltyArcAngle,
    },
  },
  {
    id: "right_penalty_arc",
    label: "Arco de penalti derecho (exterior del área)",
    world: {
      center: { x: 41.5, y: 0 }, radius: PITCH_MODEL.centerCircleRadius,
      startAngle: Math.PI - PITCH_MODEL.penaltyArcAngle, endAngle: Math.PI + PITCH_MODEL.penaltyArcAngle,
    },
  },
];
PITCH_MODEL.calibrationArcsById = Object.fromEntries(PITCH_MODEL.calibrationArcs.map((arc) => [arc.id, arc]));

PITCH_MODEL.segments = [
  [{ x: -52.5, y: -34 }, { x: 52.5, y: -34 }, { x: 52.5, y: 34 }, { x: -52.5, y: 34 }, { x: -52.5, y: -34 }],
  [{ x: 0, y: -34 }, { x: 0, y: 34 }],
  [{ x: -52.5, y: -20.16 }, { x: -36, y: -20.16 }, { x: -36, y: 20.16 }, { x: -52.5, y: 20.16 }],
  [{ x: 52.5, y: -20.16 }, { x: 36, y: -20.16 }, { x: 36, y: 20.16 }, { x: 52.5, y: 20.16 }],
  [{ x: -52.5, y: -9.16 }, { x: -47, y: -9.16 }, { x: -47, y: 9.16 }, { x: -52.5, y: 9.16 }],
  [{ x: 52.5, y: -9.16 }, { x: 47, y: -9.16 }, { x: 47, y: 9.16 }, { x: 52.5, y: 9.16 }],
];

PITCH_MODEL.sampleCircle = (center, radius, samples = 96) => (
  Array.from({ length: samples + 1 }, (_, index) => {
    const angle = (index / samples) * Math.PI * 2;
    return { x: center.x + radius * Math.cos(angle), y: center.y + radius * Math.sin(angle) };
  })
);

PITCH_MODEL.sampleArc = (center, radius, startAngle, endAngle, samples = 32) => (
  Array.from({ length: samples + 1 }, (_, index) => {
    const angle = startAngle + ((endAngle - startAngle) * index) / samples;
    return { x: center.x + radius * Math.cos(angle), y: center.y + radius * Math.sin(angle) };
  })
);
