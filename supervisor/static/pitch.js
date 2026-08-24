const PITCH_DEFAULTS = {
  lengthM: 105,
  widthM: 68,
  goalWidthM: 7.32,
  goalHalfWidthM: 3.66,
  penaltyAreaDepthM: 16.5,
  goalAreaDepthM: 5.5,
  penaltySpotDistanceM: 11,
  centreCircleRadiusM: 9.15,
};

function validPitchDimension(value, fallback) {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function buildPitchModel(lengthM = PITCH_DEFAULTS.lengthM, widthM = PITCH_DEFAULTS.widthM) {
  const length = validPitchDimension(lengthM, PITCH_DEFAULTS.lengthM);
  const width = validPitchDimension(widthM, PITCH_DEFAULTS.widthM);
  const halfLength = length / 2;
  const halfWidth = width / 2;
  const penaltyY = PITCH_DEFAULTS.goalHalfWidthM + PITCH_DEFAULTS.penaltyAreaDepthM;
  const goalAreaY = PITCH_DEFAULTS.goalHalfWidthM + PITCH_DEFAULTS.goalAreaDepthM;
  const leftPenaltyFront = -halfLength + PITCH_DEFAULTS.penaltyAreaDepthM;
  const rightPenaltyFront = halfLength - PITCH_DEFAULTS.penaltyAreaDepthM;
  const leftGoalFront = -halfLength + PITCH_DEFAULTS.goalAreaDepthM;
  const rightGoalFront = halfLength - PITCH_DEFAULTS.goalAreaDepthM;
  const leftPenaltySpot = -halfLength + PITCH_DEFAULTS.penaltySpotDistanceM;
  const rightPenaltySpot = halfLength - PITCH_DEFAULTS.penaltySpotDistanceM;
  const penaltyArcAngle = Math.acos(PITCH_DEFAULTS.goalAreaDepthM / PITCH_DEFAULTS.centreCircleRadiusM);
  const point = (x, y) => ({ x, y });
  const landmark = (id, label, x, y) => ({ id, label, world: point(x, y) });
  const line = (id, label, start, end) => ({ id, label, world: { start: point(...start), end: point(...end) } });

  const model = {
    length,
    width,
    lengthM: length,
    widthM: width,
    halfLength,
    halfWidth,
    centerCircleRadius: PITCH_DEFAULTS.centreCircleRadiusM,
    penaltyArcAngle,
    pitchPolygon: [point(-halfLength, -halfWidth), point(halfLength, -halfWidth), point(halfLength, halfWidth), point(-halfLength, halfWidth)],
    landmarks: [
      landmark("top_left_corner", "Esquina superior izquierda", -halfLength, -halfWidth),
      landmark("top_right_corner", "Esquina superior derecha", halfLength, -halfWidth),
      landmark("bottom_right_corner", "Esquina inferior derecha", halfLength, halfWidth),
      landmark("bottom_left_corner", "Esquina inferior izquierda", -halfLength, halfWidth),
      landmark("halfway_top", "Mitad de campo / banda superior", 0, -halfWidth),
      landmark("halfway_bottom", "Mitad de campo / banda inferior", 0, halfWidth),
      landmark("centre_mark", "Punto central", 0, 0),
      landmark("left_penalty_top", "Esquina superior del área izquierda", leftPenaltyFront, -penaltyY),
      landmark("left_penalty_bottom", "Esquina inferior del área izquierda", leftPenaltyFront, penaltyY),
      landmark("right_penalty_top", "Esquina superior del área derecha", rightPenaltyFront, -penaltyY),
      landmark("right_penalty_bottom", "Esquina inferior del área derecha", rightPenaltyFront, penaltyY),
      landmark("left_goal_top", "Esquina superior del área pequeña izquierda", leftGoalFront, -goalAreaY),
      landmark("left_goal_bottom", "Esquina inferior del área pequeña izquierda", leftGoalFront, goalAreaY),
      landmark("right_goal_top", "Esquina superior del área pequeña derecha", rightGoalFront, -goalAreaY),
      landmark("right_goal_bottom", "Esquina inferior del área pequeña derecha", rightGoalFront, goalAreaY),
      landmark("left_penalty_spot", "Punto de penalti izquierdo", leftPenaltySpot, 0),
      landmark("right_penalty_spot", "Punto de penalti derecho", rightPenaltySpot, 0),
      landmark("left_goal_post_top", "Base del poste superior de la portería izquierda (arriba en el diagrama)", -halfLength, -PITCH_DEFAULTS.goalHalfWidthM),
      landmark("left_goal_post_bottom", "Base del poste inferior de la portería izquierda (abajo en el diagrama)", -halfLength, PITCH_DEFAULTS.goalHalfWidthM),
      landmark("right_goal_post_top", "Base del poste superior de la portería derecha (arriba en el diagrama)", halfLength, -PITCH_DEFAULTS.goalHalfWidthM),
      landmark("right_goal_post_bottom", "Base del poste inferior de la portería derecha (abajo en el diagrama)", halfLength, PITCH_DEFAULTS.goalHalfWidthM),
      landmark("left_penalty_goal_line_top", "Intersección área de penalti / línea de gol izquierda, superior (diagrama)", -halfLength, -penaltyY),
      landmark("left_penalty_goal_line_bottom", "Intersección área de penalti / línea de gol izquierda, inferior (diagrama)", -halfLength, penaltyY),
      landmark("right_penalty_goal_line_top", "Intersección área de penalti / línea de gol derecha, superior (diagrama)", halfLength, -penaltyY),
      landmark("right_penalty_goal_line_bottom", "Intersección área de penalti / línea de gol derecha, inferior (diagrama)", halfLength, penaltyY),
      landmark("left_goal_area_goal_line_top", "Intersección área pequeña / línea de gol izquierda, superior (diagrama)", -halfLength, -goalAreaY),
      landmark("left_goal_area_goal_line_bottom", "Intersección área pequeña / línea de gol izquierda, inferior (diagrama)", -halfLength, goalAreaY),
      landmark("right_goal_area_goal_line_top", "Intersección área pequeña / línea de gol derecha, superior (diagrama)", halfLength, -goalAreaY),
      landmark("right_goal_area_goal_line_bottom", "Intersección área pequeña / línea de gol derecha, inferior (diagrama)", halfLength, goalAreaY),
      landmark("centre_circle_halfway_top", "Intersección círculo central / medio campo, superior (diagrama)", 0, -PITCH_DEFAULTS.centreCircleRadiusM),
      landmark("centre_circle_halfway_bottom", "Intersección círculo central / medio campo, inferior (diagrama)", 0, PITCH_DEFAULTS.centreCircleRadiusM),
    ],
    calibrationLines: [
      line("halfway_line", "Línea de medio campo", [0, -halfWidth], [0, halfWidth]),
      line("top_touchline", "Banda superior", [-halfLength, -halfWidth], [halfLength, -halfWidth]),
      line("bottom_touchline", "Banda inferior", [-halfLength, halfWidth], [halfLength, halfWidth]),
      line("left_goal_line", "Línea de fondo izquierda", [-halfLength, -halfWidth], [-halfLength, halfWidth]),
      line("right_goal_line", "Línea de fondo derecha", [halfLength, -halfWidth], [halfLength, halfWidth]),
      line("left_penalty_top", "Área izquierda — línea superior", [-halfLength, -penaltyY], [leftPenaltyFront, -penaltyY]),
      line("left_penalty_front", "Área izquierda — línea frontal", [leftPenaltyFront, -penaltyY], [leftPenaltyFront, penaltyY]),
      line("left_penalty_bottom", "Área izquierda — línea inferior", [leftPenaltyFront, penaltyY], [-halfLength, penaltyY]),
      line("right_penalty_top", "Área derecha — línea superior", [halfLength, -penaltyY], [rightPenaltyFront, -penaltyY]),
      line("right_penalty_front", "Área derecha — línea frontal", [rightPenaltyFront, -penaltyY], [rightPenaltyFront, penaltyY]),
      line("right_penalty_bottom", "Área derecha — línea inferior", [rightPenaltyFront, penaltyY], [halfLength, penaltyY]),
      line("left_goal_top", "Área pequeña izquierda — línea superior", [-halfLength, -goalAreaY], [leftGoalFront, -goalAreaY]),
      line("left_goal_front", "Área pequeña izquierda — línea frontal", [leftGoalFront, -goalAreaY], [leftGoalFront, goalAreaY]),
      line("left_goal_bottom", "Área pequeña izquierda — línea inferior", [leftGoalFront, goalAreaY], [-halfLength, goalAreaY]),
      line("right_goal_top", "Área pequeña derecha — línea superior", [halfLength, -goalAreaY], [rightGoalFront, -goalAreaY]),
      line("right_goal_front", "Área pequeña derecha — línea frontal", [rightGoalFront, -goalAreaY], [rightGoalFront, goalAreaY]),
      line("right_goal_bottom", "Área pequeña derecha — línea inferior", [rightGoalFront, goalAreaY], [halfLength, goalAreaY]),
    ],
    calibrationArcs: [
      { id: "centre_circle", label: "Círculo central", world: { center: point(0, 0), radius: PITCH_DEFAULTS.centreCircleRadiusM } },
      { id: "left_penalty_arc", label: "Arco de penalti izquierdo (exterior del área)", world: { center: point(leftPenaltySpot, 0), radius: PITCH_DEFAULTS.centreCircleRadiusM, startAngle: -penaltyArcAngle, endAngle: penaltyArcAngle } },
      { id: "right_penalty_arc", label: "Arco de penalti derecho (exterior del área)", world: { center: point(rightPenaltySpot, 0), radius: PITCH_DEFAULTS.centreCircleRadiusM, startAngle: Math.PI - penaltyArcAngle, endAngle: Math.PI + penaltyArcAngle } },
    ],
    segments: [
      [point(-halfLength, -halfWidth), point(halfLength, -halfWidth), point(halfLength, halfWidth), point(-halfLength, halfWidth), point(-halfLength, -halfWidth)],
      [point(0, -halfWidth), point(0, halfWidth)],
      [point(-halfLength, -penaltyY), point(leftPenaltyFront, -penaltyY), point(leftPenaltyFront, penaltyY), point(-halfLength, penaltyY)],
      [point(halfLength, -penaltyY), point(rightPenaltyFront, -penaltyY), point(rightPenaltyFront, penaltyY), point(halfLength, penaltyY)],
      [point(-halfLength, -goalAreaY), point(leftGoalFront, -goalAreaY), point(leftGoalFront, goalAreaY), point(-halfLength, goalAreaY)],
      [point(halfLength, -goalAreaY), point(rightGoalFront, -goalAreaY), point(rightGoalFront, goalAreaY), point(halfLength, goalAreaY)],
    ],
    markPoints: [point(0, 0), point(leftPenaltySpot, 0), point(rightPenaltySpot, 0)],
  };

  model.landmarksById = Object.fromEntries(model.landmarks.map((item) => [item.id, item]));
  model.calibrationLinesById = Object.fromEntries(model.calibrationLines.map((item) => [item.id, item]));
  model.calibrationArcsById = Object.fromEntries(model.calibrationArcs.map((item) => [item.id, item]));
  model.sampleCircle = (center, radius, samples = 96) => Array.from({ length: samples + 1 }, (_, index) => {
    const angle = (index / samples) * Math.PI * 2;
    return point(center.x + radius * Math.cos(angle), center.y + radius * Math.sin(angle));
  });
  model.sampleArc = (center, radius, startAngle, endAngle, samples = 32) => Array.from({ length: samples + 1 }, (_, index) => {
    const angle = startAngle + ((endAngle - startAngle) * index) / samples;
    return point(center.x + radius * Math.cos(angle), center.y + radius * Math.sin(angle));
  });
  return model;
}

let PITCH_MODEL = buildPitchModel();
