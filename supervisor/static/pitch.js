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
PITCH_MODEL.calibrationArcs = [{
  id: "centre_circle",
  label: "Círculo central",
  world: { center: { x: 0, y: 0 }, radius: PITCH_MODEL.centerCircleRadius },
}];
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
