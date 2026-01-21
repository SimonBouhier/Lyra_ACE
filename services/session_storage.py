"""
LYRA CLEAN - SESSION STORAGE SERVICE
=====================================

Gestion des sauvegardes et chargements de sessions.
Organisation: saves/{model_name}/{timestamp}_{session_id}.json

Fonctionnalités:
- Export de session avec historique complet
- Import de session dans nouvelle ou existante
- Organisation par modèle LLM utilisé
- Horodatage automatique
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class SessionExport:
    """Structure d'export de session."""
    session_id: str
    model: str
    profile: str
    created_at: str
    exported_at: str
    message_count: int
    total_tokens: int
    messages: List[Dict[str, Any]]
    trajectories: List[Dict[str, Any]]
    consciousness_adjustments: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class SessionStorage:
    """
    Service de stockage de sessions.

    Organisation des fichiers:
        saves/
        ├── mistral_latest/
        │   ├── 20250118_143052_abc123.json
        │   └── 20250118_151230_def456.json
        ├── llama3.1_8b/
        │   └── 20250117_092015_xyz789.json
        └── gpt-oss_20b/
            └── 20250116_180045_uvw012.json
    """

    def __init__(self, base_dir: str = "saves"):
        """
        Initialise le service de stockage.

        Args:
            base_dir: Répertoire racine des sauvegardes
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_model_name(self, model: str) -> str:
        """
        Convertit le nom du modèle en nom de dossier valide.

        Args:
            model: Nom du modèle (ex: "mistral:latest")

        Returns:
            Nom sanitisé (ex: "mistral_latest")
        """
        return model.replace(":", "_").replace("/", "_").replace("\\", "_")

    def _get_model_dir(self, model: str) -> Path:
        """
        Obtient le répertoire pour un modèle donné.

        Args:
            model: Nom du modèle

        Returns:
            Path du répertoire (créé si nécessaire)
        """
        model_dir = self.base_dir / self._sanitize_model_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def _generate_filename(self, session_id: str) -> str:
        """
        Génère un nom de fichier horodaté.

        Args:
            session_id: UUID de la session

        Returns:
            Nom de fichier (ex: "20250118_143052_abc123.json")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = session_id[:8] if len(session_id) > 8 else session_id
        return f"{timestamp}_{short_id}.json"

    async def export_session(
        self,
        db,  # ISpaceDB
        session_id: str,
        model: str
    ) -> Dict[str, Any]:
        """
        Exporte une session vers un fichier JSON.

        Args:
            db: Instance de la base de données
            session_id: UUID de la session à exporter
            model: Nom du modèle utilisé

        Returns:
            Dict avec path du fichier et métadonnées

        Raises:
            ValueError: Si la session n'existe pas
        """
        # Récupérer les infos de session
        async with db.connection() as conn:
            # Session metadata
            cursor = await conn.execute(
                """
                SELECT session_id, created_at, last_activity, profile,
                       message_count, total_tokens
                FROM sessions WHERE session_id = ?
                """,
                (session_id,)
            )
            session_row = await cursor.fetchone()

            if not session_row:
                raise ValueError(f"Session {session_id} not found")

            # Messages
            cursor = await conn.execute(
                """
                SELECT event_id, event_type, role, content,
                       injected_concepts, graph_weight, timestamp, latency_ms
                FROM events
                WHERE session_id = ?
                ORDER BY timestamp ASC
                """,
                (session_id,)
            )
            messages = [
                {
                    "event_id": row[0],
                    "event_type": row[1],
                    "role": row[2],
                    "content": row[3],
                    "injected_concepts": json.loads(row[4]) if row[4] else None,
                    "graph_weight": row[5],
                    "timestamp": row[6],
                    "latency_ms": row[7]
                }
                for row in await cursor.fetchall()
            ]

            # Trajectories
            cursor = await conn.execute(
                """
                SELECT trajectory_id, t_param, tau_c, rho, delta_r, kappa, timestamp
                FROM trajectories
                WHERE session_id = ?
                ORDER BY timestamp ASC
                """,
                (session_id,)
            )
            trajectories = [
                {
                    "trajectory_id": row[0],
                    "t": row[1],
                    "tau_c": row[2],
                    "rho": row[3],
                    "delta_r": row[4],
                    "kappa": row[5],
                    "timestamp": row[6]
                }
                for row in await cursor.fetchall()
            ]

            # Consciousness adjustments
            cursor = await conn.execute(
                """
                SELECT id, turn_number, metrics, adjustments, timestamp
                FROM session_adjustments
                WHERE session_id = ?
                ORDER BY turn_number ASC
                """,
                (session_id,)
            )
            adjustments = [
                {
                    "id": row[0],
                    "turn_number": row[1],
                    "metrics": json.loads(row[2]) if row[2] else None,
                    "adjustments": json.loads(row[3]) if row[3] else None,
                    "timestamp": row[4]
                }
                for row in await cursor.fetchall()
            ]

        # Créer l'export
        export = SessionExport(
            session_id=session_id,
            model=model,
            profile=session_row[3],
            created_at=datetime.fromtimestamp(session_row[1]).isoformat(),
            exported_at=datetime.now().isoformat(),
            message_count=session_row[4],
            total_tokens=session_row[5],
            messages=messages,
            trajectories=trajectories,
            consciousness_adjustments=adjustments,
            metadata={
                "lyra_version": "1.0.0",
                "export_format": "v1"
            }
        )

        # Sauvegarder le fichier
        model_dir = self._get_model_dir(model)
        filename = self._generate_filename(session_id)
        filepath = model_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(export), f, ensure_ascii=False, indent=2)

        return {
            "success": True,
            "filepath": str(filepath),
            "filename": filename,
            "model_dir": str(model_dir),
            "session_id": session_id,
            "message_count": len(messages),
            "exported_at": export.exported_at
        }

    async def import_session(
        self,
        db,  # ISpaceDB
        filepath: str,
        new_session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Importe une session depuis un fichier JSON.

        Args:
            db: Instance de la base de données
            filepath: Chemin du fichier à importer
            new_session_id: Optionnel, nouvel ID de session (sinon génère un nouveau)

        Returns:
            Dict avec session_id et statistiques d'import

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le format est invalide
        """
        import uuid

        # Charger le fichier
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Valider le format
        required_fields = ['session_id', 'model', 'profile', 'messages']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Invalid format: missing field '{field}'")

        # Générer un nouvel ID si nécessaire
        session_id = new_session_id or str(uuid.uuid4())

        async with db.connection() as conn:
            # Créer la session
            await conn.execute(
                """
                INSERT INTO sessions (session_id, created_at, last_activity, profile,
                                      message_count, total_tokens)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    time.time(),
                    time.time(),
                    data.get('profile', 'balanced'),
                    data.get('message_count', 0),
                    data.get('total_tokens', 0)
                )
            )

            # Importer les messages
            messages_imported = 0
            for msg in data.get('messages', []):
                if msg.get('role') and msg.get('content'):
                    # Utiliser le bon event_type pour que les messages soient récupérés par get_conversation_messages
                    role = msg['role']
                    event_type = f"{role}_message" if role in ['user', 'assistant'] else 'imported_message'

                    await conn.execute(
                        """
                        INSERT INTO events (session_id, event_type, role, content,
                                           injected_concepts, graph_weight, timestamp, latency_ms)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            session_id,
                            event_type,
                            role,
                            msg['content'],
                            json.dumps(msg.get('injected_concepts')) if msg.get('injected_concepts') else None,
                            msg.get('graph_weight', 0.0),
                            msg.get('timestamp', time.time()),
                            msg.get('latency_ms')
                        )
                    )
                    messages_imported += 1

            # Importer les trajectoires
            trajectories_imported = 0
            for traj in data.get('trajectories', []):
                await conn.execute(
                    """
                    INSERT INTO trajectories (session_id, t_param, tau_c, rho, delta_r, kappa, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        traj.get('t', 0.0),
                        traj.get('tau_c', 1.0),
                        traj.get('rho', 0.0),
                        traj.get('delta_r', 0.0),
                        traj.get('kappa'),
                        traj.get('timestamp', time.time())
                    )
                )
                trajectories_imported += 1

            # Importer les ajustements de conscience
            adjustments_imported = 0
            for adj in data.get('consciousness_adjustments', []):
                await conn.execute(
                    """
                    INSERT INTO session_adjustments (session_id, turn_number, metrics, adjustments, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        adj.get('turn_number', 0),
                        json.dumps(adj.get('metrics')) if adj.get('metrics') else '{}',
                        json.dumps(adj.get('adjustments')) if adj.get('adjustments') else '{}',
                        adj.get('timestamp', time.time())
                    )
                )
                adjustments_imported += 1

            await conn.commit()

        return {
            "success": True,
            "session_id": session_id,
            "original_session_id": data.get('session_id'),
            "original_model": data.get('model'),
            "messages_imported": messages_imported,
            "trajectories_imported": trajectories_imported,
            "adjustments_imported": adjustments_imported,
            "imported_at": datetime.now().isoformat()
        }

    def list_saves(self, model: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Liste les sauvegardes disponibles.

        Args:
            model: Optionnel, filtrer par modèle

        Returns:
            Liste des sauvegardes avec métadonnées
        """
        saves = []

        if model:
            # Liste pour un modèle spécifique
            model_dir = self.base_dir / self._sanitize_model_name(model)
            if model_dir.exists():
                for file in model_dir.glob("*.json"):
                    saves.append(self._get_save_info(file, model))
        else:
            # Liste tous les modèles
            for model_dir in self.base_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    for file in model_dir.glob("*.json"):
                        saves.append(self._get_save_info(file, model_name))

        # Trier par date décroissante
        saves.sort(key=lambda x: x['filename'], reverse=True)

        return saves

    def _get_save_info(self, filepath: Path, model: str) -> Dict[str, Any]:
        """
        Obtient les informations d'un fichier de sauvegarde.

        Args:
            filepath: Chemin du fichier
            model: Nom du modèle

        Returns:
            Dict avec métadonnées du fichier
        """
        stat = filepath.stat()

        # Extraire timestamp et session_id du nom de fichier
        # Format: YYYYMMDD_HHMMSS_sessionid.json
        parts = filepath.stem.split('_')
        timestamp_str = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else ""
        session_short = parts[2] if len(parts) >= 3 else filepath.stem

        # Parser le timestamp
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            timestamp_iso = timestamp.isoformat()
        except ValueError:
            timestamp_iso = None

        return {
            "filename": filepath.name,
            "filepath": str(filepath),
            "model": model,
            "session_short_id": session_short,
            "timestamp": timestamp_iso,
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 2)
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Liste les modèles ayant des sauvegardes.

        Returns:
            Liste des modèles avec nombre de sauvegardes
        """
        models = []

        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir():
                save_count = len(list(model_dir.glob("*.json")))
                if save_count > 0:
                    models.append({
                        "model": model_dir.name,
                        "save_count": save_count,
                        "path": str(model_dir)
                    })

        models.sort(key=lambda x: x['save_count'], reverse=True)
        return models

    def delete_save(self, filepath: str) -> Dict[str, Any]:
        """
        Supprime un fichier de sauvegarde.

        Args:
            filepath: Chemin du fichier à supprimer

        Returns:
            Dict avec statut de suppression
        """
        path = Path(filepath)

        if not path.exists():
            return {"success": False, "error": "File not found"}

        # Vérifier que le fichier est dans le répertoire saves
        try:
            path.relative_to(self.base_dir)
        except ValueError:
            return {"success": False, "error": "File outside saves directory"}

        path.unlink()

        return {
            "success": True,
            "deleted": str(path),
            "deleted_at": datetime.now().isoformat()
        }


# Singleton instance
_storage_instance: Optional[SessionStorage] = None


def get_session_storage(base_dir: str = "saves") -> SessionStorage:
    """
    Obtient l'instance du service de stockage (singleton).

    Args:
        base_dir: Répertoire racine des sauvegardes

    Returns:
        Instance de SessionStorage
    """
    global _storage_instance

    if _storage_instance is None:
        _storage_instance = SessionStorage(base_dir)

    return _storage_instance
